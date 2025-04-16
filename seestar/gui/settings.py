"""
Module pour la gestion des paramètres de traitement des images astronomiques.
"""


class SettingsManager:
    """
    Classe pour gérer les paramètres de traitement des images.
    """

    def __init__(self):
        """Initialise le gestionnaire de paramètres avec des valeurs par défaut."""
        # Paramètres d'alignement
        self.bayer_pattern = "GRBG"
        self.batch_size = 0
        self.reference_image_path = None

        # Paramètres des pixels chauds
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5

        # Paramètres d'empilement
        self.stacking_mode = "kappa-sigma"
        self.kappa = 2.5
        self.denoise = False
        self.remove_aligned = False

        # Mode de traitement
        self.use_queue_mode = True

        # Prévisualisation
        self.enable_preview = True
        self.apply_stretch = True

    def configure_aligner(self, aligner):
        """
        Configure un objet aligneur avec les paramètres actuels.

        Args:
            aligner (SeestarAligner): Objet aligneur à configurer
        """
        aligner.bayer_pattern = self.bayer_pattern
        aligner.batch_size = int(self.batch_size)
        aligner.reference_image_path = self.reference_image_path
        aligner.correct_hot_pixels = self.correct_hot_pixels
        aligner.hot_pixel_threshold = float(self.hot_pixel_threshold)
        aligner.neighborhood_size = int(self.neighborhood_size)

    def configure_stacker(self, stacker):
        """
        Configure un objet stacker avec les paramètres actuels.

        Args:
            stacker (ProgressiveStacker): Objet stacker à configurer
        """
        stacker.stacking_mode = self.stacking_mode
        stacker.kappa = float(self.kappa)
        stacker.batch_size = int(self.batch_size)
        stacker.denoise = self.denoise

    def configure_queued_stacker(self, queued_stacker):
        """
        Configure un objet stacker en file d'attente avec les paramètres actuels.

        Args:
            queued_stacker (SeestarQueuedStacker): Objet stacker en file d'attente à configurer
        """
        queued_stacker.stacking_mode = self.stacking_mode
        queued_stacker.kappa = float(self.kappa)
        queued_stacker.batch_size = int(self.batch_size)
        queued_stacker.correct_hot_pixels = self.correct_hot_pixels
        queued_stacker.hot_pixel_threshold = float(self.hot_pixel_threshold)
        queued_stacker.neighborhood_size = int(self.neighborhood_size)
        queued_stacker.denoise = self.denoise

    def update_from_variables(self, variables):
        """
        Met à jour les paramètres à partir des variables de l'interface utilisateur.

        Args:
            variables (dict): Dictionnaire de variables Tkinter (StringVar, BooleanVar, etc.)
        """
        # Paramètres d'alignement
        if 'reference_image_path' in variables:
            self.reference_image_path = variables['reference_image_path'].get(
            ) or None
        if 'batch_size' in variables:
            self.batch_size = int(variables['batch_size'].get())

        # Paramètres des pixels chauds
        if 'correct_hot_pixels' in variables:
            self.correct_hot_pixels = variables['correct_hot_pixels'].get()
        if 'hot_pixel_threshold' in variables:
            self.hot_pixel_threshold = float(
                variables['hot_pixel_threshold'].get())
        if 'neighborhood_size' in variables:
            self.neighborhood_size = int(variables['neighborhood_size'].get())

        # Paramètres d'empilement
        if 'stacking_mode' in variables:
            self.stacking_mode = variables['stacking_mode'].get()
        if 'kappa' in variables:
            self.kappa = float(variables['kappa'].get())
        if 'denoise' in variables:
            self.denoise = variables['denoise'].get()
        if 'remove_aligned' in variables:
            self.remove_aligned = variables['remove_aligned'].get()

        # Mode de traitement
        if 'use_queue_mode' in variables:
            self.use_queue_mode = variables['use_queue_mode'].get()

        # Prévisualisation
        if 'enable_preview' in variables:
            self.enable_preview = variables['enable_preview'].get()
        if 'apply_stretch' in variables:
            self.apply_stretch = variables['apply_stretch'].get()

    def validate_settings(self):
        """
        Valide les paramètres et effectue des ajustements si nécessaire.

        Returns:
            tuple: (valide, messages) - valide est un booléen indiquant si les paramètres sont valides,
                   messages est une liste de messages d'avertissement
        """
        valid = True
        messages = []

        # Valider la taille du voisinage (doit être impaire)
        if self.neighborhood_size % 2 == 0:
            self.neighborhood_size += 1
            messages.append(
                f"La taille du voisinage a été ajustée à {self.neighborhood_size} (doit être impaire)")

        # Valider kappa (doit être positif)
        if self.kappa <= 0:
            self.kappa = 2.5
            messages.append(
                f"La valeur kappa a été réinitialisée à {self.kappa} (doit être positive)")

        # Valider le seuil des pixels chauds (doit être positif)
        if self.hot_pixel_threshold <= 0:
            self.hot_pixel_threshold = 3.0
            messages.append(
                f"Le seuil de détection des pixels chauds a été réinitialisé à {self.hot_pixel_threshold} (doit être positif)")

        return valid, messages

    def save_settings(self, filename):
        """
        Sauvegarde les paramètres dans un fichier.

        Args:
            filename (str): Chemin du fichier de sauvegarde
        """
        import json

        # Créer un dictionnaire des paramètres
        settings = {
            # Paramètres d'alignement
            'bayer_pattern': self.bayer_pattern,
            'batch_size': self.batch_size,
            'reference_image_path': self.reference_image_path,

            # Paramètres des pixels chauds
            'correct_hot_pixels': self.correct_hot_pixels,
            'hot_pixel_threshold': self.hot_pixel_threshold,
            'neighborhood_size': self.neighborhood_size,

            # Paramètres d'empilement
            'stacking_mode': self.stacking_mode,
            'kappa': self.kappa,
            'denoise': self.denoise,
            'remove_aligned': self.remove_aligned,

            # Mode de traitement
            'use_queue_mode': self.use_queue_mode,

            # Prévisualisation
            'enable_preview': self.enable_preview,
            'apply_stretch': self.apply_stretch
        }

        # Enregistrer les paramètres dans un fichier JSON
        with open(filename, 'w') as f:
            json.dump(settings, f, indent=4)

    def load_settings(self, filename):
        """
        Charge les paramètres depuis un fichier.

        Args:
            filename (str): Chemin du fichier de sauvegarde

        Returns:
            bool: True si le chargement est réussi, False sinon
        """
        import json
        import os

        if not os.path.exists(filename):
            return False

        try:
            # Charger les paramètres depuis le fichier JSON
            with open(filename, 'r') as f:
                settings = json.load(f)

            # Mettre à jour les paramètres
            # Paramètres d'alignement
            if 'bayer_pattern' in settings:
                self.bayer_pattern = settings['bayer_pattern']
            if 'batch_size' in settings:
                self.batch_size = settings['batch_size']
            if 'reference_image_path' in settings:
                self.reference_image_path = settings['reference_image_path']

            # Paramètres des pixels chauds
            if 'correct_hot_pixels' in settings:
                self.correct_hot_pixels = settings['correct_hot_pixels']
            if 'hot_pixel_threshold' in settings:
                self.hot_pixel_threshold = settings['hot_pixel_threshold']
            if 'neighborhood_size' in settings:
                self.neighborhood_size = settings['neighborhood_size']

            # Paramètres d'empilement
            if 'stacking_mode' in settings:
                self.stacking_mode = settings['stacking_mode']
            if 'kappa' in settings:
                self.kappa = settings['kappa']
            if 'denoise' in settings:
                self.denoise = settings['denoise']
            if 'remove_aligned' in settings:
                self.remove_aligned = settings['remove_aligned']

            # Mode de traitement
            if 'use_queue_mode' in settings:
                self.use_queue_mode = settings['use_queue_mode']

            # Prévisualisation
            if 'enable_preview' in settings:
                self.enable_preview = settings['enable_preview']
            if 'apply_stretch' in settings:
                self.apply_stretch = settings['apply_stretch']

            return True
        except Exception as e:
            print(f"Erreur lors du chargement des paramètres: {e}")
            return False
