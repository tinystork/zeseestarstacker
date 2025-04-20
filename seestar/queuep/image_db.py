"""
Gestionnaire de base de données d'images pour le traitement en file d'attente.
"""
import os
from astropy.io import fits

class ImageDatabase:
    """
    Classe pour suivre les images traitées et les stacks disponibles.
    """
    def __init__(self, storage_folder):
        """
        Initialise la base de données d'images.
        
        Args:
            storage_folder (str): Dossier de stockage pour la base de données
        """
        self.storage_folder = storage_folder
        self.processed_file = os.path.join(storage_folder, "processed.txt")
        self.processed = set()
        
        # Créer le dossier de stockage s'il n'existe pas
        os.makedirs(storage_folder, exist_ok=True)
        
        # Charger la liste des fichiers déjà traités
        if os.path.isfile(self.processed_file):
            with open(self.processed_file, 'r') as f:
                self.processed = set(line.strip() for line in f)
    
    def is_processed(self, path):
        """
        Vérifie si une image a déjà été traitée.
        
        Args:
            path (str): Chemin vers l'image à vérifier
            
        Returns:
            bool: True si l'image a déjà été traitée, False sinon
        """
        return path in self.processed
    
    def mark_processed(self, path):
        """
        Marque une image comme traitée.
        
        Args:
            path (str): Chemin vers l'image à marquer
        """
        self.processed.add(path)
        with open(self.processed_file, 'a') as f:
            f.write(f"{path}\n")
    
    def get_stack(self, key):
        """
        Récupère un stack existant basé sur une clé.
        
        Args:
            key (str): Clé du stack à récupérer
            
        Returns:
            tuple: (données du stack, en-tête du stack) ou (None, None) si non trouvé
        """
        stack_path = os.path.join(self.storage_folder, f"{key}.fit")
        if os.path.isfile(stack_path):
            from seestar.core.image_processing import load_and_validate_fits
            return load_and_validate_fits(stack_path), fits.getheader(stack_path)
        return None, None
    
    def save_stack(self, stack_data, header, key):
        """
        Sauvegarde un stack dans la base de données.
        
        Args:
            stack_data (numpy.ndarray): Données du stack
            header (astropy.io.fits.Header): En-tête du stack
            key (str): Clé du stack
        """
        stack_path = os.path.join(self.storage_folder, f"{key}.fit")
        fits.writeto(stack_path, stack_data, header, overwrite=True)
        
        # Créer une prévisualisation PNG si possible
        try:
            from seestar.core.image_processing import save_preview_image
            preview_path = os.path.join(self.storage_folder, f"{key}.png")
            save_preview_image(stack_data, preview_path)
        except Exception as e:
            print(f"Erreur lors de la création de la prévisualisation: {e}")