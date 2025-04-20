"""
Fonctions de base pour le traitement d'images astronomiques.
"""
import os
import numpy as np
from astropy.io import fits
import cv2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Removed duplicated comment block and partial function definition

def load_and_validate_fits(path):
    """
    Charge un fichier FITS et vérifie qu'il s'agit bien d'une image 2D ou 3D.
    Note: This function only returns the data array. Use fits.getheader(path) separately.

    Parameters:
        path (str): Chemin vers le fichier FITS.

    Returns:
        numpy.ndarray: Données de l'image en float32.

    Raises:
        ValueError: Si l'image n'est pas en 2D ou 3D.
        FileNotFoundError: If the path does not exist.
        OSError: If there's an issue reading the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier FITS n'existe pas : {path}")
    try:
        # Use fits.getdata which is simpler if only data is needed
        data = fits.getdata(path)
        data = np.squeeze(data).astype(np.float32) # Squeeze removes dimensions of size 1
        if data.ndim not in [2, 3]:
            # Check if it's 3D but first dim is not 3 (like some cameras save)
             if data.ndim == 3 and data.shape[0] != 3:
                 # Assume it might be HxWxN, try taking the first channel if N > 1
                 if data.shape[-1] > 1 : # Check last dim size
                    print(f"Warning: FITS data at {path} has unexpected 3D shape {data.shape}. Using first channel.")
                    data = data[:,:,0] # Take first channel along last axis
                 elif data.shape[0] > 1: # Check first dim size
                     print(f"Warning: FITS data at {path} has unexpected 3D shape {data.shape}. Using first slice.")
                     data = data[0,:,:] # Take first slice along first axis
                 else: # Fallback if shape is weird e.g., 1xHxW
                     data = np.squeeze(data) # Try squeezing again

                 # Re-check dimension after potential correction
                 if data.ndim not in [2, 3]:
                    raise ValueError(f"L'image (après tentative de correction) doit être 2D (HxW) ou 3D (HxWx3 ou 3xHxW). Shape trouvée: {data.shape}")
             elif data.ndim not in [2, 3]: # Final check
                raise ValueError(f"L'image doit être 2D (HxW) ou 3D (HxWx3 ou 3xHxW). Shape trouvée: {data.shape}")
        return data
    except Exception as e:
        raise OSError(f"Erreur lors de la lecture du fichier FITS {path}: {e}")


def debayer_image(img, bayer_pattern="GRBG"):
    """
    Convertit une image brute Bayer en image RGB.

    Parameters:
        img (numpy.ndarray): Image brute (2D).
        bayer_pattern (str): Motif Bayer ("GRBG", "RGGB", "GBRG", "BGGR").

    Returns:
        numpy.ndarray: Image RGB débayerisée (float32).

    Raises:
        ValueError: Si le motif Bayer n'est pas supporté ou l'image n'est pas 2D.
    """
    if img.ndim != 2:
        raise ValueError(f"Le debayering ne s'applique qu'aux images 2D. Shape reçue: {img.shape}")

    # Normalize to uint16 for OpenCV debayering functions
    # Use np.nanmax to handle potential NaN values gracefully
    max_val = np.nanmax(img)
    min_val = np.nanmin(img)
    if max_val > min_val:
         img_uint16 = ((img - min_val) * (65535.0 / (max_val - min_val))).astype(np.uint16)
    elif max_val == min_val: # Handle constant image
         img_uint16 = np.full_like(img, 32767, dtype=np.uint16)
    else: # Handle image with NaNs only?
         img_uint16 = np.zeros_like(img, dtype=np.uint16)


    # Mapping from Bayer pattern string to OpenCV code
    bayer_codes = {
        "GRBG": cv2.COLOR_BayerGR2RGB,
        "RGGB": cv2.COLOR_BayerRG2RGB,
        "GBRG": cv2.COLOR_BayerGB2RGB,
        "BGGR": cv2.COLOR_BayerBG2RGB,
    }

    if bayer_pattern.upper() in bayer_codes:
        cv_code = bayer_codes[bayer_pattern.upper()]
        color_img = cv2.cvtColor(img_uint16, cv_code)
    else:
        raise ValueError(f"Motif Bayer '{bayer_pattern}' non supporté. Options: GRBG, RGGB, GBRG, BGGR.")

    # Convert back to float32 for consistency in processing pipeline
    return color_img.astype(np.float32)


def save_fits_image(image, output_path, header=None, overwrite=True):
    """
    Enregistre une image au format FITS (uint16).

    Parameters:
        image (numpy.ndarray): Image 2D (HxW) ou 3D (HxWx3) à enregistrer.
                               Assumed to be float32 or similar numeric type.
        output_path (str): Chemin du fichier de sortie.
        header (astropy.io.fits.Header, optional): En-tête FITS.
        overwrite (bool): Écrase le fichier existant si True.
    """
    if header is None:
        header = fits.Header()

    # Determine if image is color (HxWx3 format assumed)
    is_color = image.ndim == 3 and image.shape[-1] == 3

    if is_color:
        # FITS standard is (Channels, Height, Width) - C, Y, X
        image_fits = np.moveaxis(image, -1, 0) # HxWx3 -> 3xHxW
        header['NAXIS'] = 3
        header['NAXIS1'] = image.shape[1] # Width
        header['NAXIS2'] = image.shape[0] # Height
        header['NAXIS3'] = 3 # Channels
        # Ensure color type is indicated if not present
        if 'CTYPE3' not in header: header['CTYPE3'] = 'RGB'
    else:
        # Grayscale image (HxW)
        image_fits = image
        header['NAXIS'] = 2
        header['NAXIS1'] = image.shape[1] # Width
        header['NAXIS2'] = image.shape[0] # Height
        # Remove NAXIS3 if it exists from a previous header
        if 'NAXIS3' in header: del header['NAXIS3']
        if 'CTYPE3' in header: del header['CTYPE3']


    # Common FITS settings for saving intensity data
    header['BITPIX'] = 16 # Save as unsigned 16-bit integer
    # Using BZERO=32768 allows representing the full range of uint16
    # header['BSCALE'] = 1
    # header['BZERO'] = 32768 # Standard for unsigned integer FITS

    # Normalize image data to 0-65535 and convert to uint16
    # Use np.nanmax/min for safety
    min_val = np.nanmin(image_fits)
    max_val = np.nanmax(image_fits)

    if max_val > min_val:
        # Normalize to 0-1 float first, then scale to 0-65535
        normalized_float = (image_fits.astype(np.float32) - min_val) / (max_val - min_val)
        image_uint16 = (normalized_float * 65535.0).astype(np.uint16)
    elif max_val == min_val: # Handle constant image
        image_uint16 = np.full_like(image_fits, 0, dtype=np.uint16) # Or maybe 32767? Let's use 0.
    else: # Handle all NaN?
        image_uint16 = np.zeros_like(image_fits, dtype=np.uint16)


    # Write the FITS file
    try:
        # Create Primary HDU
        hdu = fits.PrimaryHDU(data=image_uint16, header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(output_path, overwrite=overwrite, checksum=True)
    except Exception as e:
        print(f"Error saving FITS file to {output_path}: {e}")
        # Consider re-raising or logging more formally depending on desired behavior
        raise


def save_preview_image(image, output_path, stretch=False):
    """
    Enregistre une image PNG pour la prévisualisation (normalisée 0-255).

    Parameters:
        image (numpy.ndarray): Image 2D (HxW) ou 3D (HxWx3) à sauvegarder.
        output_path (str): Chemin du fichier PNG de sortie.
        stretch (bool): Applique une normalisation automatique si True (currently ignored, simple normalization used).
                        NOTE: Stretch logic could be added here if needed using seestar.tools.Stretch
    """
    if image is None:
        print(f"Warning: Attempted to save preview for None image to {output_path}")
        return

    preview = image.copy()

    # Normalize image to 0-255 range for uint8 saving
    min_val = np.nanmin(preview)
    max_val = np.nanmax(preview)

    if max_val > min_val:
         preview_norm = ((preview - min_val) * (255.0 / (max_val - min_val)))
    elif max_val == min_val: # Handle constant image
         preview_norm = np.full_like(preview, 128) # Mid-gray
    else: # Handle all NaN?
         preview_norm = np.zeros_like(preview)

    preview_uint8 = np.clip(preview_norm, 0, 255).astype(np.uint8)


    # Handle color conversion for OpenCV saving (expects BGR)
    is_color = preview_uint8.ndim == 3 and preview_uint8.shape[-1] == 3

    if is_color:
        # Assuming input 'image' was HxWxRGB, convert RGB to BGR for cv2.imwrite
        preview_bgr = cv2.cvtColor(preview_uint8, cv2.COLOR_RGB2BGR)
    else:
        # Grayscale image, OpenCV handles it directly
        preview_bgr = preview_uint8

    # Ensure output directory exists
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except OSError as e:
        print(f"Error creating directory for preview {output_path}: {e}")
        return # Cannot save if directory fails

    # Save the image
    try:
        success = cv2.imwrite(output_path, preview_bgr)
        if not success:
             print(f"Error: cv2.imwrite failed to save preview image to {output_path}")
    except Exception as e:
        print(f"Error saving preview image to {output_path}: {e}")