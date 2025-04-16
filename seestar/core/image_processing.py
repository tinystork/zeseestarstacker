"""
Fonctions de base pour le traitement d'images astronomiques.
"""
import os
import numpy as np
from astropy.io import fits
import cv2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def load_and_validate_fits(path):
    """
    Charge et valide les fichiers FITS, en s'assurant qu'ils sont des images 2D ou 3D.

    Parameters:
        path (str): Chemin vers le fichier FITS

    Returns:
        numpy.ndarray: Données de l'image en tableau float32

    Raises:
        ValueError: Si les dimensions de l'image ne sont pas 2D ou 3D
    """
    data = fits.getdata(path)
    data = np.squeeze(data).astype(np.float32)
    if data.ndim not in [2, 3]:
        raise ValueError("L'image doit être 2D (HxW) ou 3D (HxWx3)")
    return data


def debayer_image(img, bayer_pattern="GRBG"):
    """
    Convertit une image brute Bayer en RGB.

    Parameters:
        img (numpy.ndarray): Données brutes de l'image
        bayer_pattern (str): Motif Bayer utilisé dans l'image ("GRBG" ou "RGGB")

    Returns:
        numpy.ndarray: Image RGB débayerisée

    Raises:
        ValueError: Si bayer_pattern n'est pas supporté
    """
    img_uint16 = cv2.normalize(
        img, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)

    if bayer_pattern == "GRBG":
        color_img = cv2.cvtColor(img_uint16, cv2.COLOR_BayerGR2RGB)
    elif bayer_pattern == "RGGB":
        color_img = cv2.cvtColor(img_uint16, cv2.COLOR_BayerRG2RGB)
    else:
        raise ValueError(f"Motif Bayer {bayer_pattern} non supporté")

    return color_img.astype(np.float32)


def save_fits_image(image, output_path, header=None, overwrite=True):
    """
    Enregistre une image au format FITS.

    Parameters:
        image (numpy.ndarray): Image à enregistrer (2D ou 3D)
        output_path (str): Chemin du fichier de sortie
        header (astropy.io.fits.Header): En-tête FITS optionnel
        overwrite (bool): Écraser le fichier s'il existe déjà
    """
    # Créer un en-tête par défaut si non fourni
    if header is None:
        header = fits.Header()

    # Déterminer si l'image est en couleur
    is_color = image.ndim == 3 and image.shape[2] == 3

    # Conversion pour le format FITS
    if is_color:
        # Convertir de HxWx3 à 3xHxW pour FITS standard
        image_fits = np.moveaxis(image, -1, 0)

        # Mettre à jour l'en-tête pour une image couleur
        header['NAXIS'] = 3
        header['NAXIS1'] = image.shape[1]  # Width
        header['NAXIS2'] = image.shape[0]  # Height
        header['NAXIS3'] = 3               # Color channels
        header['BITPIX'] = 16              # 16-bit
        header.set('CTYPE3', 'RGB', 'Couleurs RGB')
    else:
        image_fits = image

        # Mettre à jour l'en-tête pour une image monochrome
        header['NAXIS'] = 2
        header['NAXIS1'] = image.shape[1]  # Width
        header['NAXIS2'] = image.shape[0]  # Height
        header['BITPIX'] = 16              # 16-bit

    # Normaliser et convertir en entiers 16 bits
    image_fits = cv2.normalize(
        image_fits, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)

    # Enregistrer l'image FITS
    fits.writeto(output_path, image_fits, header, overwrite=overwrite)


def save_preview_image(image, output_path, stretch=False):
    """
    Enregistre une version PNG de l'image pour prévisualisation.

    Parameters:
        image (numpy.ndarray): Image à enregistrer (2D ou 3D)
        output_path (str): Chemin du fichier PNG de sortie
        stretch (bool): Appliquer une normalisation automatique pour améliorer la visualisation
    """
    # Copier l'image pour ne pas modifier l'original
    preview = image.copy()

    # Déterminer si l'image est en couleur
    is_color = preview.ndim == 3 and preview.shape[2] == 3

    if is_color:
        # Si l'image est déjà au format HxWx3
        preview_img = cv2.normalize(
            preview, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # OpenCV utilise BGR, donc convertir de RGB
        preview_img = cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR)
    else:
        # Image en niveaux de gris
        preview_img = cv2.normalize(
            preview, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Enregistrer l'image
    cv2.imwrite(output_path, preview_img)
