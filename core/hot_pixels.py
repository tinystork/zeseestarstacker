"""
Module pour la détection et la correction des pixels chauds dans les images astronomiques.
"""
import numpy as np
import cv2


def detect_and_correct_hot_pixels(image, threshold=3.0, neighborhood_size=5):
    """
    Détecte et corrige les pixels chauds dans une image.

    Parameters:
        image (numpy.ndarray): Image à traiter
        threshold (float): Seuil en écarts-types pour considérer un pixel comme "chaud"
        neighborhood_size (int): Taille du voisinage pour le calcul de la médiane

    Returns:
        numpy.ndarray: Image avec pixels chauds corrigés
    """
    # Vérifier si l'image est en couleur ou en niveaux de gris
    is_color = len(image.shape) == 3 and image.shape[2] == 3

    if is_color:
        # Traiter chaque canal séparément
        corrected_img = np.copy(image)
        for c in range(image.shape[2]):
            channel = image[:, :, c]

            # Calculer les statistiques locales
            mean = cv2.blur(channel, (neighborhood_size, neighborhood_size))
            mean_sq = cv2.blur(
                channel**2, (neighborhood_size, neighborhood_size))
            # Éviter les valeurs négatives
            std = np.sqrt(np.maximum(mean_sq - mean**2, 0))

            # Identifier les pixels chauds (valeurs anormalement élevées)
            hot_pixels = channel > (mean + threshold * std)

            # Appliquer une correction médiane où des pixels chauds sont détectés
            if np.any(hot_pixels):
                # Créer une version médiane de l'image
                median_filtered = cv2.medianBlur(
                    channel.astype(np.float32), neighborhood_size)

                # Remplacer uniquement les pixels chauds par leur valeur médiane
                corrected_img[:, :, c] = np.where(
                    hot_pixels, median_filtered, channel)
    else:
        # Image en niveaux de gris
        # Calculer les statistiques locales
        mean = cv2.blur(image, (neighborhood_size, neighborhood_size))
        mean_sq = cv2.blur(image**2, (neighborhood_size, neighborhood_size))
        # Éviter les valeurs négatives
        std = np.sqrt(np.maximum(mean_sq - mean**2, 0))

        # Identifier les pixels chauds
        hot_pixels = image > (mean + threshold * std)

        # Appliquer une correction médiane où des pixels chauds sont détectés
        if np.any(hot_pixels):
            median_filtered = cv2.medianBlur(
                image.astype(np.float32), neighborhood_size)
            corrected_img = np.where(hot_pixels, median_filtered, image)
        else:
            corrected_img = image

    return corrected_img
