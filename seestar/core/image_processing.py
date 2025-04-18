"""
Fonctions de base pour le traitement d'images astronomiques.
"""
import os
import numpy as np
from astropy.io import fits
import cv2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

"""
Fonctions de base pour le traitement d'images astronomiques.
"""

import os
import warnings

import cv2
import numpy as np
from astropy.io import fits

# Supprimer les avertissements liés aux futures versions
warnings.filterwarnings("ignore", category=FutureWarning)


def load_and_validate_fits(path):
    """
    Charge un fichier FITS et vérifie qu'il s'agit bien d'une image 2D ou 3D.

    Parameters:
        path (str): Chemin vers le fichier FITS.

    Returns:
        numpy.ndarray: Données de l'image en float32.

    Raises:
        ValueError: Si l'image n'est pas en 2D ou 3D.
    """
    data = fits.getdata(path)
    data = np.squeeze(data).astype(np.float32)
    if data.ndim not in [2, 3]:
        raise ValueError("L'image doit être 2D (HxW) ou 3D (HxWx3).")
    return data


def debayer_image(img, bayer_pattern="GRBG"):
    """
    Convertit une image brute Bayer en image RGB.

    Parameters:
        img (numpy.ndarray): Image brute.
        bayer_pattern (str): Motif Bayer ("GRBG" ou "RGGB").

    Returns:
        numpy.ndarray: Image RGB débayerisée.

    Raises:
        ValueError: Si le motif Bayer n'est pas supporté.
    """
    img_uint16 = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)

    if bayer_pattern == "GRBG":
        color_img = cv2.cvtColor(img_uint16, cv2.COLOR_BayerGR2RGB)
    elif bayer_pattern == "RGGB":
        color_img = cv2.cvtColor(img_uint16, cv2.COLOR_BayerRG2RGB)
    else:
        raise ValueError(f"Motif Bayer '{bayer_pattern}' non supporté.")

    return color_img.astype(np.float32)


def save_fits_image(image, output_path, header=None, overwrite=True):
    """
    Enregistre une image au format FITS.

    Parameters:
        image (numpy.ndarray): Image 2D ou 3D à enregistrer.
        output_path (str): Chemin du fichier de sortie.
        header (astropy.io.fits.Header, optional): En-tête FITS.
        overwrite (bool): Écrase le fichier existant si True.
    """
    if header is None:
        header = fits.Header()

    is_color = image.ndim == 3 and image.shape[2] == 3

    if is_color:
        # HxWx3 → 3xHxW
        image_fits = np.moveaxis(image, -1, 0)
        header['NAXIS'] = 3
        header['NAXIS1'] = image.shape[1]
        header['NAXIS2'] = image.shape[0]
        header['NAXIS3'] = 3
        header['BITPIX'] = 16
        header['CTYPE3'] = 'RGB'
    else:
        image_fits = image
        header['NAXIS'] = 2
        header['NAXIS1'] = image.shape[1]
        header['NAXIS2'] = image.shape[0]
        header['BITPIX'] = 16

    # Normaliser et convertir en uint16
    image_fits = cv2.normalize(image_fits, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
    fits.writeto(output_path, image_fits, header, overwrite=overwrite)


def save_preview_image(image, output_path, stretch=False):
    """
    Enregistre une image PNG pour la prévisualisation.

    Parameters:
        image (numpy.ndarray): Image 2D ou 3D à sauvegarder.
        output_path (str): Chemin du fichier PNG de sortie.
        stretch (bool): Applique une normalisation automatique si True.
    """
    preview = image.copy()
    is_color = preview.ndim == 3 and preview.shape[2] == 3

    if is_color:
        preview_img = cv2.normalize(preview, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        preview_img = cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR)
    else:
        preview_img = cv2.normalize(preview, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite(output_path, preview_img)
