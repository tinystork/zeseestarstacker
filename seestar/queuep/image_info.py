"""
Module pour l'extraction et la standardisation des métadonnées des images astronomiques.
"""
import time
from astropy.io import fits

class ImageInfo:
    """
    Extrait et standardise les métadonnées des images astronomiques.
    """
    def __init__(self, path):
        """
        Initialise l'objet ImageInfo en extrayant les métadonnées du fichier FITS.
        
        Args:
            path (str): Chemin vers le fichier FITS
        """
        self.path = path
        self.header = fits.getheader(path)
        
        # Extraire les métadonnées importantes
        self.camera = self.header.get('INSTRUME', 'Seestar')
        self.exposure = float(self.header.get('EXPTIME', 0))
        self.image_type = self._determine_image_type()
        self.target = self.header.get('OBJECT', '')
        self.filter = self.header.get('FILTER', 'NONE')
        self.bayer_pattern = self.header.get('BAYERPAT', 'GRBG')  # Par défaut pour Seestar
        
    def _determine_image_type(self):
        """
        Détermine le type d'image (LIGHT, DARK, FLAT, BIAS).
        
        Returns:
            str: Type d'image ('LIGHT', 'DARK', 'FLAT', 'BIAS' ou 'UNKNOWN')
        """
        # Pour Seestar, toutes les images sont probablement des lights
        img_type = self.header.get('IMAGETYP', '').upper()
        
        if img_type:
            if 'LIGHT' in img_type or 'OBJECT' in img_type:
                return 'LIGHT'
            elif 'DARK' in img_type:
                return 'DARK'
            elif 'FLAT' in img_type:
                return 'FLAT'
            elif 'BIAS' in img_type or 'ZERO' in img_type:
                return 'BIAS'
        
        # Par défaut, considérer comme LIGHT
        return 'LIGHT'
    
    @property
    def stack_key(self):
        """
        Génère une clé unique pour le stack basée sur les métadonnées.
        
        Returns:
            str: Clé unique pour le stack
        """
        # Pour Seestar, utiliser le target comme clé principale
        if self.target:
            return f"Seestar_{self.target}_{self.filter}_{self.exposure}"
        # Si pas de target, utiliser timestamp
        return f"Seestar_Light_{self.filter}_{self.exposure}_{int(time.time())}"