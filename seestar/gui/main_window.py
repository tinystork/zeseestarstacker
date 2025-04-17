"""
Module principal pour l'interface graphique de Seestar.
"""
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time
import numpy as np
import cv2
from astropy.io import fits
from queue import Queue, Empty
from seestar.core.alignment import SeestarAligner
from seestar.core.stacking import ProgressiveStacker
from seestar.queuep.queue_manager import SeestarQueuedStacker
from seestar.localization.localization import Localization
from .preview import PreviewManager
from .progress import ProgressManager
from .settings import SettingsManager
from seestar.tools.stretch import Stretch

class SeestarStackerGUI:
    """
    GUI pour l'application Seestar Stacker avec système de file d'attente.
    """
    
    def process_additional_files(self, file_list, output_folder):
        """Traite des fichiers supplémentaires en mode traditionnel."""
        try:
            # Vérifier si nous avons un stack existant
            if not hasattr(self, 'current_stack_data') or self.current_stack_data is None:
                self.update_progress("⚠️ Aucun stack existant trouvé. Impossible d'ajouter des fichiers.")
                return
                
            # Créer un dossier temporaire pour les nouveaux fichiers
            import tempfile
            import shutil
            
            temp_dir = os.path.join(output_folder, "additional_files_temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Copier les fichiers sélectionnés dans le dossier temporaire
            self.update_progress("🔄 Préparation des fichiers additionnels...")
            copied_files = []
            
            for file_path in file_list:
                dest_path = os.path.join(temp_dir, os.path.basename(file_path))
                shutil.copy2(file_path, dest_path)
                copied_files.append(dest_path)
                
            self.update_progress(f"✅ {len(copied_files)} fichiers copiés pour traitement")
            
            # Obtenir le chemin vers l'image de référence
            aligned_folder = os.path.join(output_folder, "aligned_temp")
            reference_path = os.path.join(aligned_folder, "reference_image.fit")
            
            if not os.path.exists(reference_path):
                self.update_progress("⚠️ Image de référence non trouvée. Utilisation du stack actuel comme référence.")
                # Sauvegarder le stack actuel comme nouvelle référence
                from seestar.core.image_processing import save_fits_image
                
                # Sauvegarder avec le format attendu par l'aligneur
                ref_data = np.moveaxis(self.current_stack_data, -1, 0).astype(np.uint16)  # HxWx3 to 3xHxW
                
                new_header = fits.Header()
                new_header['NAXIS'] = 3
                new_header['NAXIS1'] = self.current_stack_data.shape[1]
                new_header['NAXIS2'] = self.current_stack_data.shape[0]
                new_header['NAXIS3'] = 3
                new_header['BITPIX'] = 16
                new_header.set('CTYPE3', 'RGB', 'Couleurs RGB')
                new_header.set('REFRENCE', True, 'stacking reference')
                
                fits.writeto(reference_path, ref_data, new_header, overwrite=True)
                self.update_progress("✅ Nouvelle image de référence créée à partir du stack actuel")
            
            # Configurer l'aligneur avec cette référence
            self.aligner = SeestarAligner()  # Réinitialiser avec un nouvel aligneur
            self.aligner.reference_image_path = reference_path
            self.aligner.batch_size = len(copied_files)  # Traiter tous les fichiers en un seul lot
            self.aligner.set_progress_callback(self.update_progress)
            
            # Aligner les images
            self.update_progress("🔄 Alignement des nouvelles images sur la référence...")
            aligned_output_folder = os.path.join(output_folder, "new_aligned")
            os.makedirs(aligned_output_folder, exist_ok=True)
            
            self.aligner.align_images(temp_dir, aligned_output_folder)
            
            # Vérifier si des images ont été alignées
            aligned_files = [f for f in os.listdir(aligned_output_folder) 
                            if f.startswith('aligned_') and f.endswith('.fit')]
            
            if not aligned_files:
                self.update_progress("⚠️ Aucune image n'a pu être alignée. Abandon du traitement.")
                return
                
            self.update_progress(f"✅ {len(aligned_files)} nouvelles images alignées avec succès")
            
            # Charger les images alignées
            from seestar.core.image_processing import load_and_validate_fits
            
            self.update_progress("🔄 Chargement des images alignées...")
            aligned_images = []
            
            for file_name in aligned_files:
                try:
                    img = load_and_validate_fits(os.path.join(aligned_output_folder, file_name))
                    
                    # Convertir au format attendu si nécessaire
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = np.moveaxis(img, 0, -1)  # 3xHxW -> HxWx3
                    
                    aligned_images.append(img)
                except Exception as e:
                    self.update_progress(f"⚠️ Erreur lors du chargement de {file_name}: {e}")
            
            if not aligned_images:
                self.update_progress("⚠️ Aucune image alignée n'a pu être chargée. Abandon du traitement.")
                return
                
            # Créer un stack avec ces nouvelles images
            self.update_progress("🧮 Création d'un stack avec les nouvelles images...")
            
            if len(aligned_images) == 1:
                new_stack = aligned_images[0]
            else:
                # Empiler selon la méthode choisie
                stacking_mode = self.stacking_mode.get()
                
                if stacking_mode == "mean":
                    new_stack = np.mean(aligned_images, axis=0)
                elif stacking_mode == "median":
                    new_stack = np.median(np.stack(aligned_images, axis=0), axis=0)
                elif stacking_mode == "kappa-sigma":
                    # Calcul simplifié kappa-sigma
                    kappa = self.kappa.get()
                    stack = np.stack(aligned_images, axis=0)
                    mean = np.mean(stack, axis=0)
                    std = np.std(stack, axis=0)
                    
                    # Pour chaque canal
                    if mean.ndim == 3:
                        new_stack = np.zeros_like(mean)
                        for c in range(3):
                            sum_image = np.zeros_like(mean[:,:,c])
                            mask_sum = np.zeros_like(mean[:,:,c])
                            
                            for img in stack:
                                deviation = np.abs(img[:,:,c] - mean[:,:,c])
                                mask = deviation <= (kappa * std[:,:,c])
                                sum_image += img[:,:,c] * mask
                                mask_sum += mask
                            
                            # Éviter division par zéro
                            mask_sum = np.maximum(mask_sum, 1)
                            new_stack[:,:,c] = sum_image / mask_sum
                    else:
                        # Pour images monochrome
                        sum_image = np.zeros_like(mean)
                        mask_sum = np.zeros_like(mean)
                        
                        for img in stack:
                            deviation = np.abs(img - mean)
                            mask = deviation <= (kappa * std)
                            sum_image += img * mask
                            mask_sum += mask
                        
                        mask_sum = np.maximum(mask_sum, 1)
                        new_stack = sum_image / mask_sum
                else:
                    # Par défaut
                    new_stack = np.mean(aligned_images, axis=0)
            
            # Normaliser le nouveau stack
            new_stack = cv2.normalize(new_stack, None, 0, 65535, cv2.NORM_MINMAX)
            
            # Obtenir le nombre d'images dans le stack actuel
            if hasattr(self, 'current_stack_header') and self.current_stack_header:
                current_images = int(self.current_stack_header.get('NIMAGES', 1))
            else:
                current_images = 1
                
            new_images = len(aligned_images)
            total_images = current_images + new_images
            
            # Combiner avec le stack existant
            self.update_progress("🔄 Combinaison avec le stack existant...")
            
            weight_current = current_images / total_images
            weight_new = new_images / total_images
            
            combined_stack = (self.current_stack_data * weight_current) + (new_stack * weight_new)
            
            # Mettre à jour le stack courant
            self.current_stack_data = combined_stack
            
            # Mettre à jour l'en-tête
            if not hasattr(self, 'current_stack_header') or not self.current_stack_header:
                self.current_stack_header = fits.Header()
                
            self.current_stack_header['NIMAGES'] = total_images
            self.current_stack_header['STACKTYP'] = self.stacking_mode.get()
            
            # Sauvegarder le résultat
            from seestar.core.image_processing import save_fits_image, save_preview_image
            
            # Stack cumulatif mis à jour
            cumulative_file = os.path.join(output_folder, "stack_cumulative.fit")
            save_fits_image(self.current_stack_data, cumulative_file, self.current_stack_header)
            save_preview_image(self.current_stack_data, os.path.join(output_folder, "stack_cumulative.png"))
            
            # Stack final avec métadonnées
            color_output = os.path.join(output_folder, "stack_final_color_metadata.fit")
            ref_img_path = copied_files[0]  # Utiliser le premier des nouveaux fichiers comme référence de métadonnées
            self.save_stack_with_original_metadata(self.current_stack_data, color_output, ref_img_path)
            
            # Mettre à jour la prévisualisation
            self.update_preview(
                self.current_stack_data, 
                f"Stack après ajout de {new_images} images (total: {total_images})", 
                self.apply_stretch.get()
            )
            
            self.update_progress(f"✅ {new_images} images additionnelles combinées avec succès au stack existant")
            
            # Nettoyage si souhaité
            try:
                import shutil
                shutil.rmtree(temp_dir)
                self.update_progress("🧹 Nettoyage des fichiers temporaires effectué")
            except Exception as e:
                self.update_progress(f"⚠️ Erreur lors du nettoyage: {e}")
                
        except Exception as e:
            self.update_progress(f"❌ Erreur lors du traitement des fichiers additionnels: {e}")
            import traceback
            traceback.print_exc()
        
    def update_remaining_files(self):
        """Met à jour l'affichage du nombre de fichiers restants à traiter."""
        if hasattr(self, 'queued_stacker') and hasattr(self.queued_stacker, 'queue'):
            if self.queued_stacker.queue:
                remaining = self.queued_stacker.queue.qsize()
                if remaining == 0:
                    self.remaining_files_var.set("Aucun fichier en attente")
                elif remaining == 1:
                    self.remaining_files_var.set("1 fichier en attente")
                else:
                    self.remaining_files_var.set(f"{remaining} fichiers en attente")
            else:
                self.remaining_files_var.set("File d'attente non initialisée")
        else:
            self.remaining_files_var.set("Aucun fichier en attente")
        
    def on_window_resize(self, event):
        """Gère le redimensionnement de la fenêtre."""
        # Ne traiter que les événements de la fenêtre principale
        if event.widget == self.root:
            self.refresh_preview()


    def init_preview(self):
        """Initialise la prévisualisation avec une image de test."""
        test_img = self.preview_manager.create_test_image()
        self.update_preview(test_img, "Image de test", True)

    def __init__(self):
        """Initialise l'interface graphique de Seestar Stacker."""
        self.root = tk.Tk()

        # Initialiser la localisation (anglais par défaut)
        self.localization = Localization('en')

        # Initialiser les classes de traitement
        self.aligner = SeestarAligner()
        self.stacker = ProgressiveStacker()
        self.queued_stacker = SeestarQueuedStacker()

        # Initialiser le gestionnaire de paramètres
        self.settings = SettingsManager()

        # Variables pour les widgets
        self.init_variables()

        # Créer l'interface
        self.create_layout()

        # Initialiser les gestionnaires
        self.init_managers()

        # Initialiser la prévisualisation
#        self.init_preview()
    
        # État du traitement
        self.processing = False
        self.thread = None
        
        # Variables pour le stockage des stacks intermédiaires et finaux
        self.current_stack_data = None
        self.current_stack_header = None


    def init_preview(self):
        """Initialise la prévisualisation avec une image de test."""
        test_img = self.preview_manager.create_test_image()
        self.update_preview(test_img, "Image de test", True)

    def init_variables(self):
        """Initialise les variables pour les widgets."""
        # Chemins des dossiers
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.reference_image_path = tk.StringVar()

        # Variable pour afficher le nombre de fichiers restants
        self.remaining_files_var = tk.StringVar(value="Aucun fichier en attente")        

        # Surveillance de dossier
        self.watch_mode = tk.BooleanVar(value=False)

        # Options d'empilement
        self.stacking_mode = tk.StringVar(value="kappa-sigma")
        self.kappa = tk.DoubleVar(value=2.5)
        self.batch_size = tk.IntVar(value=0)
        self.remove_aligned = tk.BooleanVar(value=False)
        self.apply_denoise = tk.BooleanVar(value=False)

        # Options de pixels chauds
        self.correct_hot_pixels = tk.BooleanVar(value=True)
        self.hot_pixel_threshold = tk.DoubleVar(value=3.0)
        self.neighborhood_size = tk.IntVar(value=5)

        # Mode de traitement
        self.use_queue = tk.BooleanVar(value=True)
        self.use_traditional = tk.BooleanVar(value=False)

        # Options de prévisualisation
        self.apply_stretch = tk.BooleanVar(value=True)

        # Variables pour les temps
        self.remaining_time_var = tk.StringVar(value="--:--:--")
        self.elapsed_time_var = tk.StringVar(value="00:00:00")

        # Variable pour la langue
        self.language_var = tk.StringVar(value='en')

    def init_managers(self):
        """Initialise les gestionnaires de prévisualisation et de progression."""
        # Initialiser le gestionnaire de prévisualisation
        self.preview_manager = PreviewManager(
            self.preview_canvas,
            self.current_stack_label,
            self.image_info_text
        )

        # Initialiser le gestionnaire de progression
        self.progress_manager = ProgressManager(
            self.progress_bar,
            self.status_text,
            self.remaining_time_var,
            self.elapsed_time_var
        )

        # Configurer les callbacks
        self.aligner.set_progress_callback(self.update_progress)
        self.stacker.set_progress_callback(self.update_progress)
        self.queued_stacker.set_progress_callback(self.update_progress)
        self.queued_stacker.set_preview_callback(self.update_preview)

    def tr(self, key):
        """
        Raccourci pour la recherche de traduction.

        Args:
            key (str): Clé de traduction

        Returns:
            str: Texte traduit
        """
        return self.localization.get(key)

    def create_layout(self):
        """Crée la disposition des widgets de l'interface."""
        # Créer un cadre principal qui contiendra tout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Créer un cadre à gauche pour les contrôles
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Créer un cadre à droite pour la prévisualisation
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH,
                         expand=True, padx=(10, 0))

        # -----------------------
        # ZONE DE GAUCHE (CONTRÔLES)
        # -----------------------

        # Sélection de la langue
        language_frame = ttk.Frame(left_frame)
        language_frame.pack(fill=tk.X, pady=5)
        ttk.Label(language_frame, text="Language / Langue:").pack(side=tk.LEFT)
        language_combo = ttk.Combobox(
            language_frame, textvariable=self.language_var, width=15)
        language_combo['values'] = ('English', 'Français')
        language_combo.pack(side=tk.LEFT, padx=5)
        language_combo.bind('<<ComboboxSelected>>', self.change_language)

        # Dossier d'entrée
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=5)
        self.input_label = ttk.Label(input_frame, text=self.tr('input_folder'))
        self.input_label.pack(side=tk.LEFT)
        ttk.Entry(input_frame, textvariable=self.input_path, width=40).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.browse_input_button = ttk.Button(
            input_frame, text=self.tr('browse'), command=self.browse_input)
        self.browse_input_button.pack(side=tk.RIGHT)

        # Dossier de sortie
        output_frame = ttk.Frame(left_frame)
        output_frame.pack(fill=tk.X, pady=5)
        self.output_label = ttk.Label(
            output_frame, text=self.tr('output_folder'))
        self.output_label.pack(side=tk.LEFT)
        ttk.Entry(output_frame, textvariable=self.output_path, width=40).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.browse_output_button = ttk.Button(
            output_frame, text=self.tr('browse'), command=self.browse_output)
        self.browse_output_button.pack(side=tk.RIGHT)

        # Options d'empilement
        options_frame = ttk.LabelFrame(left_frame, text=self.tr('options'))
        options_frame.pack(fill=tk.X, pady=10)

        # Méthode d'empilement
        stacking_method_frame = ttk.Frame(options_frame)
        stacking_method_frame.pack(fill=tk.X, pady=5)

        self.stacking_method_label = ttk.Label(
            stacking_method_frame, text=self.tr('stacking_method'))
        self.stacking_method_label.pack(side=tk.LEFT)
        stacking_combo = ttk.Combobox(
            stacking_method_frame, textvariable=self.stacking_mode, width=15)
        stacking_combo['values'] = (
            'mean', 'median', 'kappa-sigma', 'winsorized-sigma')
        stacking_combo.pack(side=tk.LEFT, padx=5)

        # Valeur kappa
        self.kappa_label = ttk.Label(
            stacking_method_frame, text=self.tr('kappa_value'))
        self.kappa_label.pack(side=tk.LEFT, padx=10)
        ttk.Spinbox(stacking_method_frame, from_=1.0, to=5.0, increment=0.1,
                    textvariable=self.kappa, width=8).pack(side=tk.LEFT)

        # Taille des lots
        batch_frame = ttk.Frame(options_frame)
        batch_frame.pack(fill=tk.X, pady=5)

        self.batch_size_label = ttk.Label(
            batch_frame, text=self.tr('batch_size'))
        self.batch_size_label.pack(side=tk.LEFT)
        ttk.Spinbox(batch_frame, from_=0, to=500, increment=1,
                    textvariable=self.batch_size, width=8).pack(side=tk.LEFT, padx=5)

        # Mode de traitement
        processing_mode_frame = ttk.LabelFrame(
            left_frame, text="Mode de traitement")
        processing_mode_frame.pack(fill=tk.X, pady=10)

        # Option pour utiliser le mode file d'attente
        self.queue_check = ttk.Checkbutton(processing_mode_frame, text=self.tr('use_queue_mode'),
                                           variable=self.use_queue)
        self.queue_check.pack(side=tk.LEFT, padx=5, pady=5)

        # Option d'empilement traditionnel
        self.traditional_check = ttk.Checkbutton(processing_mode_frame, text=self.tr('use_traditional_mode'),
                                                 variable=self.use_traditional)
        self.traditional_check.pack(side=tk.LEFT, padx=20)
        
        # Lier les deux options
        self.use_queue.trace_add(
            "write", lambda *args: self.use_traditional.set(not self.use_queue.get()))
        self.use_traditional.trace_add(
            "write", lambda *args: self.use_queue.set(not self.use_traditional.get()))

        # Option mode surveillance
        self.watch_mode_check = ttk.Checkbutton(processing_mode_frame, 
                                               text="Mode surveillance (traiter les nouveaux fichiers)",
                                               variable=self.watch_mode)
        self.watch_mode_check.pack(side=tk.LEFT, padx=20)

        # Options d'alignement et de référence
        self.alignment_frame = ttk.LabelFrame(
            left_frame, text=self.tr('alignment'))
        self.alignment_frame.pack(fill=tk.X, pady=10)

        # Image de référence
        ref_frame = ttk.Frame(self.alignment_frame)
        ref_frame.pack(fill=tk.X, pady=5)

        self.reference_label = ttk.Label(
            ref_frame, text=self.tr('reference_image'))
        self.reference_label.pack(side=tk.LEFT, padx=5)

        ttk.Entry(ref_frame, textvariable=self.reference_image_path,
                  width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.browse_ref_button = ttk.Button(
            ref_frame, text=self.tr('browse'), command=self.browse_reference)
        self.browse_ref_button.pack(side=tk.RIGHT, padx=5)

        # Option de suppression des images après traitement
        remove_processed_frame = ttk.Frame(self.alignment_frame)
        remove_processed_frame.pack(fill=tk.X, pady=5)

        self.remove_processed_check = ttk.Checkbutton(remove_processed_frame,
                                                      text=self.tr('remove_processed'),
                                                      variable=self.remove_aligned)
        self.remove_processed_check.pack(side=tk.LEFT, padx=5)

        # Correction des pixels chauds
        self.hot_pixels_frame = ttk.LabelFrame(
            left_frame, text=self.tr('hot_pixels_correction'))
        self.hot_pixels_frame.pack(fill=tk.X, pady=10)

        hp_check_frame = ttk.Frame(self.hot_pixels_frame)
        hp_check_frame.pack(fill=tk.X, pady=5)

        self.hot_pixels_check = ttk.Checkbutton(hp_check_frame, text=self.tr('perform_hot_pixels_correction'),
                                                variable=self.correct_hot_pixels)
        self.hot_pixels_check.pack(side=tk.LEFT)

        # Paramètres des pixels chauds
        hot_params_frame = ttk.Frame(self.hot_pixels_frame)
        hot_params_frame.pack(fill=tk.X, padx=5, pady=5)

        self.hot_pixel_threshold_label = ttk.Label(
            hot_params_frame, text=self.tr('hot_pixel_threshold'))
        self.hot_pixel_threshold_label.pack(side=tk.LEFT)
        ttk.Spinbox(hot_params_frame, from_=1.0, to=10.0, increment=0.1,
                    textvariable=self.hot_pixel_threshold, width=8).pack(side=tk.LEFT, padx=5)

        self.neighborhood_size_label = ttk.Label(
            hot_params_frame, text=self.tr('neighborhood_size'))
        self.neighborhood_size_label.pack(side=tk.LEFT, padx=10)
        ttk.Spinbox(hot_params_frame, from_=3, to=15, increment=2,
                    textvariable=self.neighborhood_size, width=8).pack(side=tk.LEFT)

        # Progression
        self.progress_frame = ttk.LabelFrame(
            left_frame, text=self.tr('progress'))
        self.progress_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.progress_bar = ttk.Progressbar(self.progress_frame, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Cadre d'estimation du temps
        time_frame = ttk.Frame(self.progress_frame)
        time_frame.pack(fill=tk.X, pady=5)

        self.remaining_time_label = ttk.Label(
            time_frame, text=self.tr('estimated_time'))
        self.remaining_time_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(time_frame, textvariable=self.remaining_time_var,
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.elapsed_time_label = ttk.Label(
            time_frame, text=self.tr('elapsed_time'))
        self.elapsed_time_label.pack(side=tk.LEFT, padx=20)
        ttk.Label(time_frame, textvariable=self.elapsed_time_var,
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        # Affichage du nombre de fichiers restants
        remaining_files_frame = ttk.Frame(self.progress_frame)
        remaining_files_frame.pack(fill=tk.X, pady=5)

        ttk.Label(remaining_files_frame, text="Fichiers en attente:", 
                 font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Label(remaining_files_frame, textvariable=self.remaining_files_var,
                 font=("Arial", 9)).pack(side=tk.LEFT, padx=5)

        # Zone de texte de statut
        self.status_text = tk.Text(self.progress_frame, height=8, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)

        # Barre de défilement pour le texte de statut
        status_scrollbar = ttk.Scrollbar(
            self.status_text, orient="vertical", command=self.status_text.yview)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)

        # Boutons de contrôle
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=10)

        self.start_button = ttk.Button(control_frame, text=self.tr('start'), 
                                      command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text=self.tr('stop'), 
                                     command=self.stop_processing, 
                                     state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Bouton pour ajouter des fichiers
        self.add_files_button = ttk.Button(control_frame, 
                                          text="Ajouter des fichiers", 
                                          command=self.add_files,
                                          state=tk.DISABLED)
        self.add_files_button.pack(side=tk.RIGHT, padx=5)

        # -----------------------
        # ZONE DE DROITE (PRÉVISUALISATION)
        # -----------------------

        # Zone de prévisualisation
        self.preview_frame = ttk.LabelFrame(
            right_frame, text="Prévisualisation")
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Canvas pour afficher les images
        self.preview_canvas = tk.Canvas(
            self.preview_frame, bg="black", width=300, height=500)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Étiquette pour afficher le nom du stack actuel
        self.current_stack_label = ttk.Label(
            self.preview_frame, text="Aucun stack en cours")
        self.current_stack_label.pack(pady=5)

        # Information sur l'image
        self.image_info_text = tk.Text(
            self.preview_frame, height=4, wrap=tk.WORD)
        self.image_info_text.pack(fill=tk.X, expand=False, pady=5)
        self.image_info_text.insert(
            tk.END, "Information sur l'image: en attente de traitement")
        self.image_info_text.config(state=tk.DISABLED)  # Lecture seule

        # Option d'étirement pour la prévisualisation
        stretch_frame = ttk.Frame(self.preview_frame)
        stretch_frame.pack(fill=tk.X, pady=5)

        self.stretch_check = ttk.Checkbutton(stretch_frame, text="Appliquer un étirement d'affichage",
                                             variable=self.apply_stretch,
                                             command=self.refresh_preview)
        self.stretch_check.pack(side=tk.LEFT, padx=5)

    def add_files(self):
        """Ouvre une boîte de dialogue pour ajouter des fichiers à la file d'attente."""
        if not self.processing:
            return
        
        # Ouvrir la boîte de dialogue pour sélectionner des fichiers
        files = filedialog.askopenfilenames(
            title="Sélectionner des fichiers à ajouter",
            filetypes=[("FITS files", "*.fit;*.fits")]
        )
        
        if not files:
            return  # Aucun fichier sélectionné
        
        # Convertir le tuple de fichiers en liste
        file_list = list(files)
        
        # Traitement différent selon le mode
        if self.use_queue.get():
            # Mode file d'attente
            
            # Initialiser une file vide si nécessaire
            if not hasattr(self.queued_stacker, 'queue') or self.queued_stacker.queue is None:
                self.queued_stacker.queue = Queue()
                self.queued_stacker.processed_files = set()
                self.update_progress("🔄 Initialisation d'une file d'attente pour les nouveaux fichiers")
            
            # Ajouter les fichiers à la file d'attente
            self.queued_stacker.add_files(file_list)
            self.update_progress(f"📋 {len(file_list)} fichiers ajoutés à la file d'attente.")
            
            # Mettre à jour l'affichage du nombre de fichiers restants
            self.update_remaining_files()
        else:
            # Mode traditionnel - utiliser la méthode dédiée
            self.update_progress(f"📋 {len(file_list)} fichiers sélectionnés pour traitement en mode traditionnel.")
            
            # Vérifier si le traitement en mode traditionnel est terminé
            if not hasattr(self, 'current_stack_data') or self.current_stack_data is None:
                self.update_progress("⚠️ Aucun stack actif. Traitement traditionnel terminé ou non démarré.")
                messagebox.showwarning(
                    "Ajout impossible",
                    "Impossible d'ajouter des fichiers car aucun stack actif n'a été trouvé.\n"
                    "Assurez-vous que le traitement traditionnel est en cours."
                )
                return
            
            # Démarrer un thread pour traiter ces fichiers
            process_thread = threading.Thread(
                target=self.process_additional_files,
                args=(file_list, self.output_path.get())
            )
            process_thread.daemon = True
            process_thread.start()  

    def process_additional_files(self, file_list, output_folder):
        """Traite des fichiers supplémentaires en mode traditionnel."""
        try:
            # Pour chaque fichier sélectionné
            for i, file_path in enumerate(file_list):
                if not self.processing:  # Arrêter si l'utilisateur a demandé l'arrêt
                    break
                    
                file_name = os.path.basename(file_path)
                self.update_progress(f"🔄 Traitement supplémentaire de {file_name}... ({i+1}/{len(file_list)})")
                
                # Charger l'image
                from seestar.core.image_processing import load_and_validate_fits
                img_data = load_and_validate_fits(file_path)
                
                # Obtenir le dossier d'entrée à partir du chemin du fichier
                input_folder = os.path.dirname(file_path)
                
                # Appliquer le même traitement que dans run_enhanced_traditional_stacking
                # mais seulement pour cette image
                # ...
                
                # Pour simplifier, nous allons juste mettre à jour la prévisualisation
                # avec cette nouvelle image
                if hasattr(self, 'preview_callback') and self.preview_callback:
                    self.update_preview(img_data, f"Image ajoutée: {file_name}", self.apply_stretch.get())
                    
        except Exception as e:
            self.update_progress(f"❌ Erreur lors du traitement des fichiers supplémentaires: {e}")

    def initialize_empty_queue(self):
        """Initialise une file d'attente vide pour permettre l'ajout de fichiers."""
        if not hasattr(self.queued_stacker, 'queue') or self.queued_stacker.queue is None:
            self.queued_stacker.queue = Queue()
            self.queued_stacker.processed_files = set()
            return True
        return False

    def refresh_preview(self):
        """Actualise l'aperçu avec la dernière image stockée."""
        if hasattr(self, 'queued_stacker') and hasattr(self.queued_stacker, 'image_db') and self.queued_stacker.image_db:
            # Récupérer le dernier stack si possible
            try:
                output_folder = self.output_path.get()
                if output_folder and os.path.exists(output_folder):
                    for key in sorted(os.listdir(output_folder), reverse=True):
                        if key.endswith('.fit') and not key.startswith('reference'):
                            stack_key = os.path.splitext(key)[0]
                            stack_data, _ = self.queued_stacker.image_db.get_stack(stack_key)
                            if stack_data is not None:
                                self.update_preview(stack_data, stack_key, self.apply_stretch.get())
                                return
            except Exception as e:
                print(f"Erreur lors de l'actualisation de la prévisualisation: {e}")
        # Si on a un stack courant en mémoire (pour le mode traditionnel amélioré)
        elif self.current_stack_data is not None:
            self.update_preview(self.current_stack_data, "Stack actuel", self.apply_stretch.get())
        elif hasattr(self, 'input_path') and self.input_path.get():
            # Si aucun stack n'est disponible mais qu'un dossier d'entrée est choisi,
            # essayer d'afficher la première image
            try:
                input_folder = self.input_path.get()
                files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))]
                if files:
                    first_image = os.path.join(input_folder, files[0])
                    from seestar.core.image_processing import load_and_validate_fits, debayer_image
                    img_data = load_and_validate_fits(first_image)
                    
                    # Appliquer le debayering si nécessaire
                    if img_data.ndim == 2:
                        from astropy.io import fits
                        try:
                            header = fits.getheader(first_image)
                            bayer_pattern = header.get('BAYERPAT', 'GRBG')
                        except:
                            bayer_pattern = 'GRBG'  # Motif par défaut
                        
                        img_data = debayer_image(img_data, bayer_pattern)
                    
                    self.update_preview(img_data, os.path.basename(first_image), self.apply_stretch.get())
            except Exception as e:
                print(f"Erreur lors du chargement de la première image: {e}")

    def update_preview(self, image_data, stack_name=None, apply_stretch=None):
        """
        Met à jour l'aperçu avec l'image fournie.

        Args:
            image_data (numpy.ndarray): Données de l'image
            stack_name (str, optional): Nom du stack
            apply_stretch (bool, optional): Appliquer un étirement automatique à l'image
        """
        if apply_stretch is None:
            apply_stretch = self.apply_stretch.get()

        # Utiliser le gestionnaire de prévisualisation
        self.preview_manager.update_preview(
        image_data, stack_name, apply_stretch)

    def browse_input(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier d'entrée."""
        folder = filedialog.askdirectory()
        if folder:
            self.input_path.set(folder)
            # Essayer d'afficher la première image ou le dernier stack
            self.refresh_preview()

    def browse_output(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier de sortie."""
        folder = filedialog.askdirectory()
        if folder:
            self.output_path.set(folder)

    def browse_reference(self):
        """Ouvre une boîte de dialogue pour sélectionner l'image de référence."""
        file = filedialog.askopenfilename(
            filetypes=[("FITS files", "*.fit;*.fits")])
        if file:
            self.reference_image_path.set(file)

    def update_progress(self, message, progress=None):
        """
        Met à jour la barre de progression et le texte de statut.

        Args:
            message (str): Message à afficher dans la zone de statut
            progress (float, optional): Valeur de progression (0-100)
        """
        # Utiliser le gestionnaire de progression
        self.progress_manager.update_progress(message, progress)

    def change_language(self, event=None):
        """Change l'interface à la langue sélectionnée."""
        selected = self.language_var.get()

        if selected == 'English':
            self.localization.set_language('en')
        elif selected == 'Français':
            self.localization.set_language('fr')

        # Mettre à jour tous les éléments de l'interface avec la nouvelle langue
        self.update_ui_language()

    def update_ui_language(self):
        """Met à jour tous les éléments de l'interface avec la langue actuelle."""
        # Mise à jour du titre de la fenêtre
        self.root.title(self.tr('title'))

        # Mise à jour des étiquettes
        self.input_label.config(text=self.tr('input_folder'))
        self.output_label.config(text=self.tr('output_folder'))
        self.browse_input_button.config(text=self.tr('browse'))
        self.browse_output_button.config(text=self.tr('browse'))
        self.browse_ref_button.config(text=self.tr('browse'))

        # Mise à jour des options
        self.alignment_frame.config(text=self.tr('alignment'))
        self.reference_label.config(text=self.tr('reference_image'))

        # Mise à jour des étiquettes d'options
        self.stacking_method_label.config(text=self.tr('stacking_method'))
        self.kappa_label.config(text=self.tr('kappa_value'))
        self.batch_size_label.config(text=self.tr('batch_size'))

        # Mise à jour de la section de progression
        self.progress_frame.config(text=self.tr('progress'))
        self.remaining_time_label.config(text=self.tr('estimated_time'))
        self.elapsed_time_label.config(text=self.tr('elapsed_time'))

        # Mise à jour des boutons
        self.start_button.config(text=self.tr('start'))
        self.stop_button.config(text=self.tr('stop'))

        # Mise à jour de la section de correction des pixels chauds
        self.hot_pixels_frame.config(text=self.tr('hot_pixels_correction'))
        self.hot_pixels_check.config(
            text=self.tr('perform_hot_pixels_correction'))
        self.hot_pixel_threshold_label.config(
            text=self.tr('hot_pixel_threshold'))
        self.neighborhood_size_label.config(text=self.tr('neighborhood_size'))

        # Mise à jour des options de mode de traitement
        self.queue_check.config(text=self.tr('use_queue_mode'))
        self.traditional_check.config(text=self.tr('use_traditional_mode'))
        self.remove_processed_check.config(text=self.tr('remove_processed'))

    def start_periodic_update(self):
        """Démarre la mise à jour périodique du nombre de fichiers restants."""
        if self.processing:
            self.update_remaining_files()
            # Programmer la prochaine mise à jour dans 1 seconde
            self.root.after(1000, self.start_periodic_update)

    def start_processing(self):
        """Démarre le traitement des images."""
        input_folder = self.input_path.get()
        output_folder = self.output_path.get()
        
        if not input_folder or not output_folder:
            messagebox.showerror(self.tr('error'), self.tr('select_folders'))
            return

        # Vérifier si le mode surveillance est activé avec le mode traditionnel
        if self.watch_mode.get() and not self.use_queue.get():
            messagebox.showwarning(
                "Mode incompatible",
                "Le mode surveillance ne peut être utilisé qu'avec le mode file d'attente.\n"
                "Veuillez activer le mode file d'attente ou désactiver le mode surveillance."
            )
            return
        
        # Mettre à jour les paramètres depuis les variables
        self.settings.update_from_variables({
            'reference_image_path': self.reference_image_path,
            'batch_size': self.batch_size,
            'correct_hot_pixels': self.correct_hot_pixels,
            'hot_pixel_threshold': self.hot_pixel_threshold,
            'neighborhood_size': self.neighborhood_size,
            'stacking_mode': self.stacking_mode,
            'kappa': self.kappa,
            'remove_aligned': self.remove_aligned,
            'use_queue_mode': self.use_queue,
            'apply_stretch': self.apply_stretch,
            'watch_mode': self.watch_mode  # Nouvelle variable
        })
        # Valider les paramètres
        valid, messages = self.settings.validate_settings()
        if messages:
            for message in messages:
                self.update_progress(f"⚠️ {message}")

        # Configurer les objets de traitement
        self.settings.configure_aligner(self.aligner)
        self.settings.configure_stacker(self.stacker)
        self.settings.configure_queued_stacker(self.queued_stacker)


        # Désactiver le bouton de démarrage et activer le bouton d'arrêt
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.processing = True

        # Activer le bouton d'ajout de fichiers dans tous les cas
        self.add_files_button.config(state=tk.NORMAL)

        # Démarrer le minuteur
        self.progress_manager.start_timer()

        # Réinitialiser le stack courant
        self.current_stack_data = None
        self.current_stack_header = None

        # Lancer le traitement dans un thread séparé
        self.thread = threading.Thread(
            target=self.run_processing, args=(input_folder, output_folder))
        self.thread.daemon = True
        self.thread.start()
        
       # Mettre à jour une première fois au démarrage
        self.root.after(500, self.update_remaining_files)  # Attendre 500ms pour que la queue soit initialisée

        # Démarrer les mises à jour périodiques
        self.root.after(500, self.start_periodic_update)

    def stop_processing(self):
        """Arrête le traitement en cours."""
        if self.processing:
            if self.use_queue.get():
                self.queued_stacker.stop()
            else:
                self.aligner.stop_processing = True
                self.stacker.stop_processing = True

            self.update_progress(self.tr('stop_requested'))


    def run_processing(self, input_folder, output_folder):
        """
        Exécute le processus d'empilement selon le mode choisi.
        """
        try:
            if self.use_queue.get():
                # Mode file d'attente
                self.update_progress(self.tr('queue_start'))

                # Initialiser la base de données
                self.queued_stacker.initialize_database(output_folder)

                # Démarrer le traitement
                stacking_thread = self.queued_stacker.start_processing(
                    input_folder, output_folder)

                # Attendre la fin du traitement (sera interrompu si stop est appelé)
                if stacking_thread:
                    stacking_thread.join()

                self.update_progress(self.tr('queue_completed'))

            else:
                # Mode traditionnel amélioré avec visualisation en temps réel
                self.run_enhanced_traditional_stacking(input_folder, output_folder)

        except Exception as e:
            self.update_progress(f"❌ {self.tr('error')}: {e}")
        finally:
            self.processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

            # Arrêter le minuteur
            self.progress_manager.stop_timer()
    def run_enhanced_traditional_stacking(self, input_folder, output_folder):
        """
        Exécute un processus d'empilement traditionnel amélioré avec visualisation en temps réel.
        Cette fonction garde l'alignement de base mais ajoute des fonctionnalités de visualisation
        et de combinaison progressive des stacks.

        Args:
            input_folder (str): Dossier contenant les images brutes
            output_folder (str): Dossier de sortie pour les images empilées
        """
        try:
            # Créer les dossiers nécessaires
            os.makedirs(output_folder, exist_ok=True)
            aligned_folder = os.path.join(output_folder, "aligned_temp")
            os.makedirs(aligned_folder, exist_ok=True)

            # Récupérer les fichiers d'entrée
            all_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))]
            if not all_files:
                self.update_progress("❌ Aucun fichier .fit/.fits trouvé")
                return

            # Estimer la taille des lots si auto
            if self.batch_size.get() <= 0:
                from seestar.core.utils import estimate_batch_size
                sample_path = os.path.join(input_folder, all_files[0])
                self.batch_size.set(estimate_batch_size(sample_path))
                self.update_progress(f"🧠 Taille de lot automatique estimée : {self.batch_size.get()}")

            batch_size = self.batch_size.get()
            total_files = len(all_files)
            self.update_progress(f"🔍 {total_files} images trouvées à traiter en lots de {batch_size}")

            # Étape 1: Obtenir une image de référence
            # (on utilise la méthode de l'aligneur pour sélectionner la référence)
            self.update_progress("🔍 Recherche de l'image de référence...")
            
            # Si l'utilisateur a spécifié une image de référence, l'utiliser
            if self.reference_image_path.get():
                reference_files = None  # L'aligneur utilisera directement reference_image_path
            else:
                # Sinon utiliser jusqu'à 50 images pour trouver une bonne référence
                reference_files = all_files[:min(50, len(all_files))]
                
            self.aligner.stop_processing = False
            reference_folder = self.aligner.align_images(
                input_folder, 
                aligned_folder, 
                specific_files=reference_files
            )
            
            # Vérifier que l'image de référence existe
            reference_image_path = os.path.join(reference_folder, "reference_image.fit")
            if not os.path.exists(reference_image_path):
                self.update_progress("❌ Échec de la création de l'image de référence")
                return
            
            self.update_progress(f"⭐ Image de référence créée : {reference_image_path}")
            
            # Nettoyer le dossier d'alignement temporaire
            for f in os.listdir(aligned_folder):
                if f != "reference_image.fit" and f != "unaligned":
                    os.remove(os.path.join(aligned_folder, f))
            
            # Traiter les images par lots
            start_time = time.time()
            stack_count = 0
            batch_count = (total_files + batch_size - 1) // batch_size  # Ceil division
            
            for batch_idx in range(batch_count):
                if self.aligner.stop_processing or self.stacker.stop_processing:
                    self.update_progress("⛔ Traitement arrêté par l'utilisateur.")
                    break
                    
                # Calculer les indices du lot actuel
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_files)
                current_files = all_files[batch_start:batch_end]
                
                self.update_progress(
                    f"🚀 Traitement du lot {batch_idx + 1}/{batch_count} "
                    f"(images {batch_start+1} à {batch_end}/{total_files})...",
                    batch_idx * 100.0 / batch_count
                )
                
                # Étape 2: Aligner les images du lot sur l'image de référence
                self.update_progress(f"📐 Alignement des images du lot {batch_idx + 1}...")
                
                # Utiliser l'image de référence comme référence explicite pour tous les lots
                original_ref_path = self.aligner.reference_image_path
                self.aligner.reference_image_path = reference_image_path
                
                # Aligner les images du lot
                self.aligner.stop_processing = False
                self.aligner.align_images(
                    input_folder, 
                    aligned_folder, 
                    specific_files=current_files
                )
                
                # Restaurer la référence d'origine
                self.aligner.reference_image_path = original_ref_path
                
                # Vérifier si l'alignement a été arrêté
                if self.aligner.stop_processing:
                    self.update_progress("⛔ Alignement arrêté par l'utilisateur.")
                    break
                
                # Étape 3: Empiler les images alignées du lot
                self.update_progress(f"🧮 Empilement des images alignées du lot {batch_idx + 1}...")
                
                # Trouver les fichiers alignés (format aligned_XXXX.fit)
                aligned_files = [f for f in os.listdir(aligned_folder) 
                               if f.startswith('aligned_') and f.endswith('.fit')]
                
                if not aligned_files:
                    self.update_progress(f"⚠️ Aucune image alignée trouvée pour le lot {batch_idx + 1}")
                    continue
                
                # Empiler les images alignées du lot
                stack_file = os.path.join(output_folder, f"stack_batch_{batch_idx+1:03d}.fit")
                
                # Charger et empiler les images alignées
                batch_stack_data, batch_stack_header = self._stack_aligned_images(
                    aligned_folder, aligned_files, stack_file
                )
                
                if batch_stack_data is None:
                    self.update_progress(f"⚠️ Échec de l'empilement du lot {batch_idx + 1}")
                    continue
                
                stack_count += 1
                
                # Mettre à jour la prévisualisation avec le résultat du lot
                self.update_preview(
                    batch_stack_data, 
                    f"Stack du lot {batch_idx + 1}", 
                    self.apply_stretch.get()
                )
                
                # Étape 4: Combiner avec le stack cumulatif si ce n'est pas le premier lot
                if self.current_stack_data is None:
                    # Premier lot, initialiser le stack cumulatif
                    self.current_stack_data = batch_stack_data
                    self.current_stack_header = batch_stack_header
                else:
                    # Combiner le stack du lot avec le stack cumulatif
                    self.update_progress(f"🔄 Fusion avec le stack cumulatif...")
                    self._combine_with_current_stack(batch_stack_data, batch_stack_header)
                
                # Mettre à jour la prévisualisation avec le stack cumulatif
                self.update_preview(
                    self.current_stack_data, 
                    f"Stack cumulatif ({batch_idx + 1}/{batch_count} lots)", 
                    self.apply_stretch.get()
                )
                
                # Enregistrer le stack cumulatif
                cumulative_file = os.path.join(output_folder, f"stack_cumulative.fit")
                from seestar.core.image_processing import save_fits_image, save_preview_image
                save_fits_image(self.current_stack_data, cumulative_file, self.current_stack_header)
                save_preview_image(self.current_stack_data, os.path.join(output_folder, "stack_cumulative.png"))
                # Sauvegarder aussi une version avec les métadonnées originales
                if self.current_stack_data is not None and all_files:
                    ref_img_path = os.path.join(input_folder, all_files[0])  # Prend la première image comme référence
                    color_output = os.path.join(output_folder, "stack_final_color_metadata.fit")
                    self.save_stack_with_original_metadata(self.current_stack_data, color_output, ref_img_path)
                
                # Suppression des images alignées traitées si demandé
                if self.remove_aligned.get():
                    for f in aligned_files:
                        try:
                            os.remove(os.path.join(aligned_folder, f))
                        except Exception as e:
                            self.update_progress(f"⚠️ Impossible de supprimer {f}: {e}")
                    
                    self.update_progress(f"🧹 Images alignées du lot {batch_idx + 1} supprimées")
            
            # Appliquer le débruitage au stack final si demandé
            if self.apply_denoise.get() and self.current_stack_data is not None:
                self.update_progress("🧹 Application du débruitage au stack final...")
                try:
                    from seestar.core.utils import apply_denoise
                    self.current_stack_data = apply_denoise(self.current_stack_data, strength=5)
                    
                    # Mise à jour du header
                    self.current_stack_header['DENOISED'] = True
                    
                    # Sauvegarder la version débruitée
                    final_file = os.path.join(output_folder, "stack_final_denoised.fit")
                    from seestar.core.image_processing import save_fits_image, save_preview_image
                    save_fits_image(self.current_stack_data, final_file, self.current_stack_header)
                    save_preview_image(self.current_stack_data, os.path.join(output_folder, "stack_final_denoised.png"))
                    
                    # Mise à jour de la prévisualisation
                    self.update_preview(
                        self.current_stack_data, 
                        "Stack final débruité", 
                        self.apply_stretch.get()
                    )
                except Exception as e:
                    self.update_progress(f"⚠️ Échec du débruitage: {e}")
            
            # Rapport final
            if stack_count > 0:
                self.update_progress(f"✅ Traitement terminé : {stack_count} lots empilés")
            else:
                self.update_progress("⚠️ Aucun stack n'a été créé")
            
        except Exception as e:
            self.update_progress(f"❌ Erreur lors du traitement: {e}")
            import traceback
            traceback.print_exc()
    
    def _stack_aligned_images(self, aligned_folder, aligned_files, output_file):
        """
        Empile les images alignées et sauvegarde le résultat.
        
        Args:
            aligned_folder (str): Dossier contenant les images alignées
            aligned_files (list): Liste des fichiers à empiler
            output_file (str): Chemin du fichier de sortie
            
        Returns:
            tuple: (stack_data, stack_header) ou (None, None) si échec
        """
        try:
            from seestar.core.image_processing import load_and_validate_fits, save_fits_image, save_preview_image
            
            # Charger toutes les images du lot
            images = []
            headers = []
            
            for file in aligned_files:
                try:
                    file_path = os.path.join(aligned_folder, file)
                    img = load_and_validate_fits(file_path)
                    
                    # Si l'image est 3D avec la première dimension comme canaux (3xHxW), 
                    # convertir en HxWx3
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = np.moveaxis(img, 0, -1)
                    
                    images.append(img)
                    headers.append(fits.getheader(file_path))
                except Exception as e:
                    self.update_progress(f"⚠️ Erreur lors du chargement de {file}: {e}")
            
            if not images:
                self.update_progress(f"❌ Aucune image valide trouvée à empiler")
                return None, None
            
            # Empiler les images selon la méthode choisie
            stacking_mode = self.stacking_mode.get()
            self.update_progress(f"🧮 Empilement avec la méthode '{stacking_mode}'...")
            
            if stacking_mode == "mean":
                stacked_image = np.mean(images, axis=0)
            elif stacking_mode == "median":
                stacked_image = np.median(np.stack(images, axis=0), axis=0)
            elif stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                kappa = self.kappa.get()
                # Convertir la liste d'images en un tableau 3D/4D
                stack = np.stack(images, axis=0)
                
                # Calculer la moyenne et l'écart-type
                mean = np.mean(stack, axis=0)
                std = np.std(stack, axis=0)
                
                if stacking_mode == "kappa-sigma":
                    # Pour kappa-sigma, on crée des masques pour chaque image
                    sum_image = np.zeros_like(mean)
                    mask_sum = np.zeros_like(mean)
                    
                    for img in stack:
                        deviation = np.abs(img - mean)
                        mask = deviation <= (kappa * std)
                        sum_image += img * mask
                        mask_sum += mask
                    
                    # Éviter la division par zéro
                    mask_sum = np.maximum(mask_sum, 1)
                    stacked_image = sum_image / mask_sum
                    
                elif stacking_mode == "winsorized-sigma":
                    # Pour winsorized-sigma, on remplace les valeurs extrêmes
                    upper_bound = mean + kappa * std
                    lower_bound = mean - kappa * std
                    
                    # Appliquer les limites à chaque image
                    clipped_stack = np.clip(stack, lower_bound, upper_bound)
                    
                    # Calculer la moyenne des images recadrées
                    stacked_image = np.mean(clipped_stack, axis=0)
            else:
                # Fallback sur la moyenne si méthode non reconnue
                self.update_progress(f"⚠️ Méthode d'empilement '{stacking_mode}' non reconnue, utilisation de 'mean'")
                stacked_image = np.mean(images, axis=0)
            
            # Créer un en-tête FITS
            stack_header = fits.Header()
            stack_header['STACKED'] = True
            stack_header['STACKTYP'] = stacking_mode
            stack_header['NIMAGES'] = len(images)
            
            if stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                stack_header['KAPPA'] = self.kappa.get()
            
            # Conserver les métadonnées importantes de la première image
            important_keys = ['INSTRUME', 'EXPTIME', 'FILTER', 'OBJECT', 'DATE-OBS']
            for key in important_keys:
                if headers and key in headers[0]:
                    stack_header[key] = headers[0][key]
            
            # Sauvegarder le stack
            save_fits_image(stacked_image, output_file, header=stack_header)
            
            # Sauvegarder une prévisualisation PNG
            preview_file = os.path.splitext(output_file)[0] + ".png"
            save_preview_image(stacked_image, preview_file)
            
            self.update_progress(f"✅ Stack créé et sauvegardé: {output_file}")
            
            return stacked_image, stack_header
            
        except Exception as e:
            self.update_progress(f"❌ Erreur lors de l'empilement: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _combine_with_current_stack(self, new_stack_data, new_stack_header):
        """
        Combine le nouveau stack avec le stack cumulatif actuel.
        
        Args:
            new_stack_data (numpy.ndarray): Données du nouveau stack
            new_stack_header (astropy.io.fits.Header): En-tête du nouveau stack
        """
        try:
            # Vérifier que les dimensions sont compatibles
            if new_stack_data.shape != self.current_stack_data.shape:
                self.update_progress(f"⚠️ Dimensions incompatibles: stack actuel {self.current_stack_data.shape}, nouveau {new_stack_data.shape}")
                
                # Essayer de redimensionner (à implémenter si nécessaire)
                return
            
            # Obtenir le nombre d'images dans chaque stack
            current_images = int(self.current_stack_header.get('NIMAGES', 1))
            new_images = int(new_stack_header.get('NIMAGES', 1))
            total_images = current_images + new_images
            
            # Vérifier que la même méthode d'empilement est utilisée
            current_method = self.current_stack_header.get('STACKTYP', 'mean')
            new_method = new_stack_header.get('STACKTYP', 'mean')
            
            if current_method != new_method:
                self.update_progress(f"⚠️ Méthodes d'empilement différentes: actuel {current_method}, nouveau {new_method}")
                # Continuer quand même, mais en notant la différence
            
            # Combiner en pondérant par le nombre d'images
            weight_current = current_images / total_images
            weight_new = new_images / total_images
            
            self.current_stack_data = (self.current_stack_data * weight_current) + (new_stack_data * weight_new)
            
            # Mettre à jour l'en-tête
            self.current_stack_header['NIMAGES'] = total_images
            self.current_stack_header['STACKTYP'] = current_method  # Conserver la méthode d'origine
            
        except Exception as e:
            self.update_progress(f"❌ Erreur lors de la combinaison des stacks: {e}")
            import traceback
            traceback.print_exc()
 
 
    def save_stack_with_original_metadata(self,stacked_image, output_path, original_path=None):
        """
        Sauvegarde une image empilée en couleur avec les métadonnées de l'image originale.
        
        Args:
            stacked_image (numpy.ndarray): Image empilée RGB (HxWx3)
            output_path (str): Chemin du fichier de sortie
            original_path (str): Chemin d'une image originale pour récupérer les métadonnées
        """
        # Créer un en-tête de base
        new_header = fits.Header()
        
        # Si l'image est en RGB (HxWx3), la convertir au format 3xHxW pour FITS
        if stacked_image.ndim == 3 and stacked_image.shape[2] == 3:
            # Conversion de HxWx3 à 3xHxW
            fits_image = np.moveaxis(stacked_image, -1, 0)
            new_header['NAXIS'] = 3
            new_header['NAXIS1'] = stacked_image.shape[1]  # Largeur
            new_header['NAXIS2'] = stacked_image.shape[0]  # Hauteur
            new_header['NAXIS3'] = 3  # 3 canaux
            new_header['CTYPE3'] = 'RGB'
        else:
            fits_image = stacked_image
            new_header['NAXIS'] = 2
            new_header['NAXIS1'] = stacked_image.shape[1]
            new_header['NAXIS2'] = stacked_image.shape[0]
        
        new_header['BITPIX'] = 16
        
        # Si une image originale est fournie, récupérer ses métadonnées
        if original_path and os.path.exists(original_path):
            try:
                orig_header = fits.getheader(original_path)
                
                # Liste des clés à conserver de l'original
                keys_to_preserve = [
                    'TELESCOP', 'INSTRUME', 'EXPTIME', 'FILTER',
                    'RA', 'DEC', 'FOCALLEN', 'APERTURE', 'SITELONG', 'SITELAT',
                    'CCD-TEMP', 'GAIN', 'XPIXSZ', 'YPIXSZ', 'FOCUSPOS',
                    'OBJECT', 'DATE-OBS', 'CREATOR', 'PRODUCER', 'PROGRAM'
                ]
                
                # Copier les métadonnées de l'original
                for key in keys_to_preserve:
                    if key in orig_header:
                        new_header[key] = orig_header[key]
            except Exception as e:
                print(f"Erreur lors de la récupération des métadonnées originales: {e}")
        
        # Ajouter les informations d'empilement
        new_header['STACKED'] = True
        new_header['STACKTYP'] = 'color-stack'
        new_header['BAYERPAT'] = 'N/A'  # Indiquer que ce n'est plus une image Bayer
        
        # Sauvegarder l'image
        fits_image = cv2.normalize(fits_image, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
        fits.writeto(output_path, fits_image, new_header, overwrite=True)
        print(f"Image empilée en couleur sauvegardée: {output_path}")
    
    
    def run(self):
        """Lance l'interface graphique."""
        self.root.mainloop()