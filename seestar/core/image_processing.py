# --- START OF FILE seestar/core/image_processing.py ---
"""
Fonctions de base pour le traitement d'images astronomiques.
"""
import os
import numpy as np
from astropy.io import fits
import cv2
import warnings
from astropy.io.fits.verify import VerifyWarning
from PIL import Image, ImageEnhance, ImageFilter # Added PIL imports for enhanced stretch

warnings.filterwarnings("ignore", category=FutureWarning)


def load_and_validate_fits(path):
    """
    Charge un fichier FITS et vérifie qu'il s'agit bien d'une image 2D ou 3D.
    Renvoie les données sous forme de tableau float32 normalisé 0-1.

    Parameters:
        path (str): Chemin vers le fichier FITS.

    Returns:
        numpy.ndarray: Données de l'image en float32 (0.0-1.0).
                       Retourne None si le chargement échoue.
    """
    if not os.path.exists(path):
        print(f"Error: FITS file not found: {path}")
        return None
    try:
        with fits.open(path) as hdul:
            # Find the first HDU with image data
            hdu = None
            for h in hdul:
                if h.data is not None and h.is_image:
                    hdu = h
                    break
            if hdu is None:
                print(f"Error: No valid image data found in FITS file: {path}")
                return None

            data = hdu.data
            # Handle potential byte order issues for non-native endianness
            if data is not None and not data.dtype.isnative:
                 data = data.byteswap().newbyteorder()

            # Convert to float32 for processing
            data = data.astype(np.float32)

            # Remove dimensions of size 1 (e.g., [1, H, W] -> [H, W])
            original_shape = data.shape
            data = np.squeeze(data)
            squeezed_shape = data.shape

            # Handle potential transposition for common camera formats (e.g., ZWO)
            # If 3D and first dim is 3 (likely color channels), move to last axis
            if data.ndim == 3 and data.shape[0] == 3 and data.shape[1] > 3 and data.shape[2] > 3:
                print(f"Detected FITS shape {original_shape} likely (C, H, W), transposing to (H, W, C).")
                data = np.transpose(data, (1, 2, 0)) # C,H,W -> H,W,C
            # If 3D and last dim is 3 (already H,W,C) - do nothing
            elif data.ndim == 3 and data.shape[-1] == 3:
                 pass # Already H,W,C
            # If 2D (Grayscale H,W) - do nothing
            elif data.ndim == 2:
                 pass
            # Other unexpected 3D+ shapes
            elif data.ndim >= 3:
                 print(f"Warning: FITS data at {path} has unexpected {data.ndim}D shape {squeezed_shape} after squeeze. Attempting to use first 2D slice/channel.")
                 # Try taking the first slice/channel, hoping it's usable image data
                 # This is a guess and might not be correct for all formats.
                 if data.shape[0] > 1 and data.shape[1] > 1 : data = data[0,...] # Take first slice along first axis
                 elif data.shape[-1] > 1 and data.shape[-2] > 1: data = data[...,0] # Take first slice along last axis
                 # Re-squeeze in case the slice revealed a single dimension
                 data = np.squeeze(data)
                 if data.ndim not in [2, 3] or (data.ndim == 3 and data.shape[-1] != 3):
                      print(f"Error: Could not resolve FITS shape {original_shape} into 2D or 3D (HxWx3).")
                      return None # Give up if still not 2D or HxWx3 color

            # Final check for valid dimensions (2D or HxWx3)
            if data.ndim not in [2, 3] or (data.ndim == 3 and data.shape[-1] != 3):
                 print(f"Error: Image must be 2D (HxW) or 3D (HxWx3). Final shape after processing: {data.shape}")
                 return None

            # Normalize data to 0.0 - 1.0 range, handling NaNs/Infs
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)

            if max_val > min_val:
                 data = (data - min_val) / (max_val - min_val)
                 data = np.clip(data, 0.0, 1.0) # Ensure data is within [0, 1]
                 data = np.nan_to_num(data, nan=0.0) # Replace any remaining NaNs with 0
            elif max_val == min_val: # Handle constant image
                 # Return a constant image scaled 0-1 if possible, else 0
                 constant_value = min_val / 65535.0 if min_val >=0 else 0.0 # Simple scaling guess
                 data = np.full_like(data, constant_value, dtype=np.float32)
                 data = np.clip(data, 0.0, 1.0)
            else: # Handle all NaN/Inf image
                 print(f"Warning: Image data in {path} seems to be all NaN or Inf.")
                 data = np.zeros_like(data, dtype=np.float32) # Return black image

            return data.astype(np.float32) # Ensure float32 output

    except FileNotFoundError: # Specific catch
         print(f"Error: File not found at: {path}")
         return None
    except Exception as e:
        print(f"Error reading or processing FITS file {path}: {e}")
        import traceback
        traceback.print_exc(limit=2)
        return None


def debayer_image(img, bayer_pattern="GRBG"):
    """
    Convertit une image brute Bayer (normalisée 0-1 float32) en image RGB (0-1 float32).

    Parameters:
        img (numpy.ndarray): Image brute (2D, float32, 0-1).
        bayer_pattern (str): Motif Bayer ("GRBG", "RGGB", "GBRG", "BGGR").

    Returns:
        numpy.ndarray: Image RGB débayerisée (float32, 0-1).

    Raises:
        ValueError: Si le motif Bayer n'est pas supporté ou l'image n'est pas 2D.
    """
    if img.ndim != 2:
        raise ValueError(f"Le debayering ne s'applique qu'aux images 2D. Shape reçue: {img.shape}")

    # Scale to uint16 for OpenCV debayering functions
    img_uint16 = (np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)

    # Mapping from Bayer pattern string to OpenCV code
    bayer_codes = {
        "GRBG": cv2.COLOR_BayerGR2RGB,
        "RGGB": cv2.COLOR_BayerRG2RGB,
        "GBRG": cv2.COLOR_BayerGB2RGB,
        "BGGR": cv2.COLOR_BayerBG2RGB,
    }

    bayer_pattern_upper = bayer_pattern.upper()
    if bayer_pattern_upper in bayer_codes:
        cv_code = bayer_codes[bayer_pattern_upper]
        try:
             # Note: OpenCV Bayer functions output BGR format
             color_img_bgr_uint16 = cv2.cvtColor(img_uint16, cv_code)
             # Convert BGR to RGB
             color_img_rgb_uint16 = cv2.cvtColor(color_img_bgr_uint16, cv2.COLOR_BGR2RGB)
        except cv2.error as cv_err:
             raise ValueError(f"OpenCV error during debayering ({bayer_pattern_upper}): {cv_err}")
    else:
        raise ValueError(f"Motif Bayer '{bayer_pattern}' non supporté. Options: GRBG, RGGB, GBRG, BGGR.")

    # Convert back to float32 (0-1 range)
    return color_img_rgb_uint16.astype(np.float32) / 65535.0


def save_fits_image(image, output_path, header=None, overwrite=True):
    """
    Enregistre une image (normalisée 0-1 float32) au format FITS (uint16).

    Parameters:
        image (numpy.ndarray): Image 2D (HxW) ou 3D (HxWx3) à enregistrer (float32, 0-1).
        output_path (str): Chemin du fichier de sortie.
        header (astropy.io.fits.Header, optional): En-tête FITS.
        overwrite (bool): Écrase le fichier existant si True.
    """
    if image is None:
        print(f"Error: Cannot save None image to {output_path}")
        return
    if not isinstance(image, np.ndarray):
        print(f"Error: Input for save_fits_image must be a numpy array, got {type(image)}")
        return

    # Ensure header is valid or create a new one
    final_header = fits.Header()
    if header is not None and isinstance(header, fits.Header):
        # Copy existing header but clear structural keywords we will redefine
        keywords_to_remove = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                             'EXTEND', 'BSCALE', 'BZERO']
        temp_header = header.copy()
        for key in keywords_to_remove:
            if key in temp_header:
                del temp_header[key]
        # Update final_header with the remaining keywords
        final_header.update(temp_header)
    elif header is not None:
        print("Warning: Provided header is not valid astropy.io.fits.Header. Creating new one.")

    # Determine if image is color (HxWx3 format assumed)
    is_color = image.ndim == 3 and image.shape[-1] == 3

    # Prepare data: Ensure float32, clip 0-1, scale to uint16, handle axis order
    image_float32 = image.astype(np.float32)  # Ensure float type
    image_clipped = np.clip(image_float32, 0.0, 1.0)
    image_uint16 = (image_clipped * 65535.0).astype(np.uint16)

    if is_color:
        # FITS standard is (Channels, Height, Width) - C, Y, X
        # Input is HxWx3, convert to 3xHxW for saving
        image_to_save = np.moveaxis(image_uint16, -1, 0)
        final_header['NAXIS'] = 3
        final_header['NAXIS1'] = image.shape[1]  # Width
        final_header['NAXIS2'] = image.shape[0]  # Height
        final_header['NAXIS3'] = 3  # Channels
        if 'CTYPE3' not in final_header:
            final_header['CTYPE3'] = ('RGB', 'Color Format')
    else:  # Grayscale image (HxW)
        image_to_save = image_uint16
        final_header['NAXIS'] = 2
        final_header['NAXIS1'] = image.shape[1]  # Width
        final_header['NAXIS2'] = image.shape[0]  # Height
        # Remove potential leftover 3rd axis keys
        if 'NAXIS3' in final_header:
            del final_header['NAXIS3']
        if 'CTYPE3' in final_header:
            del final_header['CTYPE3']

    # Common FITS settings for saving intensity data as uint16
    final_header['BITPIX'] = 16
    final_header['BSCALE'] = 1
    final_header['BZERO'] = 32768  # Standard for unsigned integer FITS

    # --- Write the FITS file WITH warning suppression ---
    try:
        hdu = fits.PrimaryHDU(data=image_to_save, header=final_header)
        hdul = fits.HDUList([hdu])

        # --- Add warning suppression context ---
        with warnings.catch_warnings():
            # Filter out the specific warning about keyword length/characters
            warnings.filterwarnings(
                'ignore',
                category=VerifyWarning,
                message="Keyword name.*is greater than 8 characters.*"
            )
            warnings.filterwarnings(
                'ignore',
                category=VerifyWarning,
                message="Keyword name.*contains characters not allowed.*"
            )
            # The actual writing happens within the suppressed context
            hdul.writeto(output_path, overwrite=overwrite, checksum=True)
        # --- End warning suppression context ---
    except Exception as e:
        print(f"An error occurred while saving the FITS file: {e}")

        
def save_preview_image(image, output_path, apply_stretch=False, enhanced_stretch=False, color_balance=(1.0, 1.0, 1.0)):
    """
    Enregistre une image PNG/JPG pour la prévisualisation (normalisée 0-255 uint8).
    Applique potentiellement un étirement et une balance des blancs pour l'aperçu.

    Parameters:
        image (numpy.ndarray): Image 2D (HxW) ou 3D (HxWx3) à sauvegarder (float32, 0-1 range).
        output_path (str): Chemin du fichier PNG/JPG de sortie.
        apply_stretch (bool): Applique un étirement automatique si True.
        enhanced_stretch (bool): Si True et apply_stretch est True, utilise l'étirement amélioré.
        color_balance (tuple): Facteurs (R, G, B) pour la balance des blancs de l'aperçu.
    """
    if image is None:
        print(f"Warning: Attempted to save preview for None image to {output_path}")
        return

    preview = image.astype(np.float32).copy() # Work on a float32 copy

    # --- Apply Color Balance (only for color images) ---
    if preview.ndim == 3 and preview.shape[-1] == 3:
        try:
            # Import within function to potentially resolve circular dependency issues
            from seestar.tools.stretch import ColorCorrection
            preview = ColorCorrection.white_balance(preview, *color_balance)
        except ImportError:
             print("Warning: Could not import ColorCorrection for preview balance.")
        except Exception as wb_err:
             print(f"Warning: Failed to apply color balance to preview: {wb_err}")


    # --- Apply Stretch ---
    if apply_stretch:
        try:
            # Determine which stretch function to use
            if enhanced_stretch:
                from seestar.tools.stretch import apply_enhanced_stretch
                preview = apply_enhanced_stretch(preview) # Returns 0-1 float
            else:
                # Use a simple linear stretch based on percentiles for basic preview
                from seestar.tools.stretch import apply_auto_stretch, StretchPresets
                bp, wp = apply_auto_stretch(preview)
                preview = StretchPresets.linear(preview, bp, wp)

            preview = np.clip(preview, 0.0, 1.0) # Ensure stretch stays in 0-1

        except ImportError:
            print("Warning: Could not import stretch tools for preview generation.")
        except Exception as e:
            print(f"Error during preview stretch: {e}, saving non-stretched version.")
            # Fallback: use the original (potentially color-balanced) data without stretch

    # --- Convert to uint8 for saving ---
    # Scale float (0-1) to uint8 (0-255)
    preview_uint8 = (np.clip(preview, 0.0, 1.0) * 255.0).astype(np.uint8)

    # --- Handle color conversion for OpenCV saving (expects BGR) ---
    if preview_uint8.ndim == 3 and preview_uint8.shape[-1] == 3:
        # Input 'preview' was HxWxRGB (float), preview_uint8 is HxWxRGB (uint8)
        # Convert RGB to BGR for cv2.imwrite
        preview_bgr = cv2.cvtColor(preview_uint8, cv2.COLOR_RGB2BGR)
    elif preview_uint8.ndim == 2:
        # Grayscale image, OpenCV handles it directly
        preview_bgr = preview_uint8
    else:
        print(f"Error: Cannot save preview with shape {preview_uint8.shape}")
        return

    # --- Ensure output directory exists ---
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: # Only create if path includes a directory
             os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory for preview {output_path}: {e}")
        return # Cannot save if directory fails

    # --- Save the image ---
    try:
        success = cv2.imwrite(output_path, preview_bgr)
        if not success:
             print(f"Error: cv2.imwrite failed to save preview image to {output_path}")
    except Exception as e:
        print(f"Error saving preview image to {output_path}: {e}")
# --- END OF FILE seestar/core/image_processing.py ---