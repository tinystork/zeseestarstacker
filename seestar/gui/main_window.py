"""
Module principal pour l'interface graphique de Seestar.
"""
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time

from seestar.core.alignment import SeestarAligner
from seestar.core.stacking import ProgressiveStacker
from seestar.queuep.queue_manager import SeestarQueuedStacker
from seestar.localization.localization import Localization
from .preview import PreviewManager
from .progress import ProgressManager
from .settings import SettingsManager


class SeestarStackerGUI:
    """
    GUI pour l'application Seestar Stacker avec système de file d'attente.
    """

    def on_window_resize(self, event):
        """Gère le redimensionnement de la fenêtre."""
        # Ne traiter que les événements de la fenêtre principale
        if event.widget == self.root:
            self.refresh_preview()


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
        self.init_preview()
    
        # État du traitement
        self.processing = False
        self.thread = None


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
        self.queue_check = ttk.Checkbutton(processing_mode_frame, text="Utiliser le mode file d'attente",
                                           variable=self.use_queue)
        self.queue_check.pack(side=tk.LEFT, padx=5, pady=5)

        # Option d'empilement traditionnel
        self.traditional_check = ttk.Checkbutton(processing_mode_frame, text="Utiliser l'empilement traditionnel",
                                                 variable=self.use_traditional)
        self.traditional_check.pack(side=tk.LEFT, padx=20)

        # Lier les deux options
        self.use_queue.trace_add(
            "write", lambda *args: self.use_traditional.set(not self.use_queue.get()))
        self.use_traditional.trace_add(
            "write", lambda *args: self.use_queue.set(not self.use_traditional.get()))

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
                                                      text="Supprimer les images traitées",
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

        self.start_button = ttk.Button(control_frame, text=self.tr(
            'start'), command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text=self.tr(
            'stop'), command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

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

    def refresh_preview(self):
        """Actualise l'aperçu avec la dernière image stockée."""
        if hasattr(self, 'queued_stacker') and self.queued_stacker.image_db:
            # Récupérer le dernier stack si possible
            try:
                output_folder = self.output_path.get()
                if output_folder and os.path.exists(output_folder):
                    for key in sorted(os.listdir(output_folder), reverse=True):
                        if key.endswith('.fit') and not key.startswith('reference'):
                            stack_key = os.path.splitext(key)[0]
                            stack_data, _ = self.queued_stacker.image_db.get_stack(
                                stack_key)
                            if stack_data is not None:
                                self.update_preview(
                                    stack_data, stack_key, self.apply_stretch.get())
                                break
            except Exception as e:
                print(
                    f"Erreur lors de l'actualisation de la prévisualisation: {e}")

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

    def start_processing(self):
        """Démarre le traitement des images."""
        input_folder = self.input_path.get()
        output_folder = self.output_path.get()

        if not input_folder or not output_folder:
            messagebox.showerror(self.tr('error'), self.tr('select_folders'))
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
            'apply_stretch': self.apply_stretch
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

        # Démarrer le minuteur
        self.progress_manager.start_timer()
        self.remaining_time_var.set(self.tr('calculating'))

        # Lancer le traitement dans un thread séparé
        self.thread = threading.Thread(
            target=self.run_processing, args=(input_folder, output_folder))
        self.thread.daemon = True
        self.thread.start()

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

        Args:
            input_folder (str): Dossier contenant les images brutes
            output_folder (str): Dossier de sortie pour les images empilées
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
                # Mode traditionnel en deux étapes
                aligned_folder = input_folder

                # Étape 1: Alignement des images
                self.update_progress(self.tr('alignment_start'))
                aligned_folder = self.aligner.align_images(input_folder)

                if self.aligner.stop_processing:
                    raise Exception(self.tr('processing_stopped'))

                self.update_progress(
                    self.tr('using_aligned_folder').format(aligned_folder))

                # Étape 2: Empilement des images alignées
                self.update_progress(self.tr('progressive_stacking_start'))

                # Créer le dossier de sortie final s'il n'existe pas
                os.makedirs(output_folder, exist_ok=True)

                # Traiter les images alignées par lots
                stack_files = self.stacker.process_aligned_images(
                    aligned_folder,
                    output_folder,
                    remove_aligned=self.remove_aligned.get()
                )

                if not stack_files:
                    self.update_progress(f"⚠️ {self.tr('no_stacks_created')}")
                else:
                    self.update_progress(
                        f"✅ {self.tr('processing_completed')} {len(stack_files)} {self.tr('stacks_created')}")

        except Exception as e:
            self.update_progress(f"❌ {self.tr('error')}: {e}")
        finally:
            self.processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

            # Arrêter le minuteur
            self.progress_manager.stop_timer()

    def run(self):
        """Lance l'interface graphique."""
        self.root.mainloop()