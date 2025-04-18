"""
Fonctions utilitaires pour le traitement d'images astronomiques.
"""
import numpy as np
import cv2
from .image_processing import load_and_validate_fits


def estimate_batch_size(sample_image_path=None, available_memory_percentage=70):
    """
    Estime la taille de lot optimale en fonction de la mémoire disponible.

    Parameters:
        sample_image_path: Chemin vers une image exemple pour estimer la taille mémoire
        available_memory_percentage: Pourcentage de la mémoire disponible à utiliser (0-100)

    Returns:
        int: Taille de lot estimée, au moins 3 et au plus 50
    """
    try:
        import psutil

        # Obtenir la mémoire disponible (en octets)
        available_memory = psutil.virtual_memory().available

        # N'utiliser qu'un pourcentage de la mémoire disponible
        usable_memory = available_memory * (available_memory_percentage / 100)

        # Estimer la taille d'une image
        if sample_image_path:
            img = load_and_validate_fits(sample_image_path)
            # Une image traitée peut prendre jusqu'à 4x plus de mémoire 
            # (versions originale, débayerisée, normalisée, alignée, plus les tampons intermédiaires)
            single_image_size = img.nbytes * 4
            
            # Pour les images couleur, prendre en compte le facteur de conversion 
            # de dimensions durant le traitement
            if img.ndim == 2:
                # Une image 2D sera convertie en 3D (3 canaux) durant le traitement
                single_image_size = single_image_size * 3
        else:
            # Estimation prudente pour une image de taille moyenne (2000x2000 pixels, 3 canaux, float32)
            single_image_size = 2000 * 2000 * 3 * 4  # environ 48 Mo

        # Facteur de sécurité pour tenir compte des opérations supplémentaires
        safety_factor = 2.5
        
        # Calculer combien d'images peuvent tenir en mémoire
        estimated_batch = int(usable_memory / (single_image_size * safety_factor))
        
        # Limites raisonnables pour la taille des lots (au moins 3, au plus 50)
        estimated_batch = max(3, min(50, estimated_batch))

        print(f"Mémoire disponible: {available_memory / (1024**3):.2f} Go")
        print(f"Taille estimée par image: {single_image_size / (1024**2):.2f} Mo")
        print(f"Taille de lot estimée: {estimated_batch}")

        return estimated_batch
    except Exception as e:
        print(f"Erreur lors de l'estimation de la taille de lot: {e}")
        import traceback
        traceback.print_exc()
        return 10  # Valeur par défaut en cas d'erreur

def apply_denoise(image, strength=1):
    """
    Applique le débruitage Non-Local Means d'OpenCV sur une image 2D (mono) ou 3D (RGB).
    Le paramètre 'strength' contrôle la réduction de bruit (valeurs recommandées : 3 à 15).

    Parameters:
        image (numpy.ndarray): Image à débruiter
        strength (int): Force du débruitage

    Returns:
        numpy.ndarray: Image débruitée
    """
    if image.ndim == 2:
        # Image monochrome
        image_uint8 = cv2.normalize(
            image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(
            image_uint8, None, h=strength, templateWindowSize=7, searchWindowSize=21)
        return denoised.astype(np.float32)

    elif image.ndim == 3 and image.shape[2] == 3:
        # Image couleur RGB
        image_uint8 = cv2.normalize(
            image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoisingColored(image_uint8, None, h=strength, hColor=strength,
                                                   templateWindowSize=7, searchWindowSize=21)
        return denoised.astype(np.float32)

    else:
        raise ValueError("L'image doit être 2D (grayscale) ou 3D (RGB)")
