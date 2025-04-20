"""
Fonctions utilitaires pour le traitement d'images astronomiques.
"""
import numpy as np
import cv2
import os # Added for exists check
import traceback # Added for better error reporting
from .image_processing import load_and_validate_fits

# Try importing psutil, but make it optional
try:
    import psutil
    _psutil_available = True
except ImportError:
    _psutil_available = False
    print("Optional dependency 'psutil' not found. Automatic batch size estimation may be limited.")


def estimate_batch_size(sample_image_path=None, available_memory_percentage=70):
    """
    Estime la taille de lot optimale en fonction de la mémoire disponible.

    Parameters:
        sample_image_path: Chemin vers une image exemple pour estimer la taille mémoire
        available_memory_percentage: Pourcentage de la mémoire disponible à utiliser (0-100)

    Returns:
        int: Taille de lot estimée, au moins 3 et au plus 50
    """
    # Default batch size if estimation fails
    default_batch_size = 10

    if not _psutil_available:
        print("psutil not available, using default batch size:", default_batch_size)
        return default_batch_size

    try:
        # Obtenir la mémoire disponible (en octets)
        mem = psutil.virtual_memory()
        available_memory = mem.available

        # N'utiliser qu'un pourcentage de la mémoire disponible
        usable_memory = available_memory * (available_memory_percentage / 100.0)

        # Estimer la taille d'une image
        single_image_size = 0
        img_ndim = 2 # Assume 2D initially
        if sample_image_path and os.path.exists(sample_image_path):
            try:
                img = load_and_validate_fits(sample_image_path)
                img_ndim = img.ndim
                # An image might involve multiple copies in memory during processing
                # (original, float32, debayered/color, normalized, aligned buffer)
                # Factor of 4-6 is a reasonable estimate for intermediate steps.
                # Let's use 5.
                memory_factor = 5
                single_image_size = img.nbytes * memory_factor

                # If starting as 2D, it will become 3 channels (HxWx3) after debayer
                if img_ndim == 2:
                    single_image_size = single_image_size * 3

            except Exception as img_e:
                 print(f"Warning: Could not load sample image {sample_image_path} for size estimation: {img_e}")
                 single_image_size = 0 # Fallback
        else:
            print("Warning: No valid sample image path provided for size estimation.")
            single_image_size = 0 # Fallback


        # Fallback estimation if image loading failed or no path provided
        if single_image_size == 0:
            # Conservative estimate for a ~4MP color image (float32) + overhead
            # 2000 * 2000 * 3 channels * 4 bytes/float * 5x overhead = 240 MB
            print("Using fallback image size estimation (approx. 4MP color image).")
            single_image_size = 2000 * 2000 * 3 * 4 * 5


        # Safety factor for other system usage and Python overhead (increase slightly)
        safety_factor = 3.0 # Increased from 2.5

        if single_image_size <= 0: # Should not happen, but safeguard
             print("Error: Calculated image size is zero or negative.")
             return default_batch_size

        # Calculer combien d'images peuvent tenir en mémoire
        estimated_batch = int(usable_memory / (single_image_size * safety_factor))

        # Limites raisonnables pour la taille des lots (au moins 3, au plus 50)
        estimated_batch = max(3, min(50, estimated_batch))

        print(f"Mémoire disponible: {available_memory / (1024**3):.2f} Go "
              f"(Utilisable: {usable_memory / (1024**3):.2f} Go)")
        print(f"Taille estimée par image (avec overhead): {single_image_size / (1024**2):.2f} Mo")
        print(f"Taille de lot estimée: {estimated_batch}")

        return estimated_batch

    except Exception as e:
        print(f"Erreur lors de l'estimation de la taille de lot: {e}")
        traceback.print_exc()
        print(f"Utilisation de la taille de lot par défaut : {default_batch_size}")
        return default_batch_size

def apply_denoise(image, strength=1):
    """
    Applique le débruitage Non-Local Means d'OpenCV sur une image 2D (mono) ou 3D (RGB).
    Le paramètre 'strength' (h) contrôle la réduction de bruit (valeurs recommandées : 3 à 15).

    Parameters:
        image (numpy.ndarray): Image à débruiter (assumed HxW or HxWx3, numeric type).
        strength (int): Force du débruitage (paramètre 'h' de fastNlMeansDenoising*).

    Returns:
        numpy.ndarray: Image débruitée (float32).
    """
    if image is None:
        print("Warning: apply_denoise received a None image.")
        return None

    # Normalize image to uint8 (0-255) as required by OpenCV denoising functions
    min_val = np.nanmin(image)
    max_val = np.nanmax(image)
    if max_val > min_val:
         image_norm = (image.astype(np.float32) - min_val) / (max_val - min_val)
         image_uint8 = (image_norm * 255.0).astype(np.uint8)
    elif max_val == min_val:
         image_uint8 = np.full_like(image, 128, dtype=np.uint8)
    else:
         image_uint8 = np.zeros_like(image, dtype=np.uint8)


    try:
        if image_uint8.ndim == 2:
            # Image monochrome
            print(f"Applying fastNlMeansDenoising with h={strength}")
            denoised_uint8 = cv2.fastNlMeansDenoising(
                image_uint8, None, h=float(strength), templateWindowSize=7, searchWindowSize=21)

        elif image_uint8.ndim == 3 and image_uint8.shape[-1] == 3:
             # Image couleur RGB (ensure input is uint8 HxWx3)
             # Need to check if input 'image' was BGR or RGB originally?
             # Assuming input 'image' was HxWxRGB float. image_uint8 is HxWxRGB uint8.
             # fastNlMeansDenoisingColored expects BGR uint8.
             print(f"Applying fastNlMeansDenoisingColored with h={strength}, hColor={strength}")
             image_bgr_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
             denoised_bgr_uint8 = cv2.fastNlMeansDenoisingColored(
                 image_bgr_uint8, None, h=float(strength), hColor=float(strength),
                 templateWindowSize=7, searchWindowSize=21)
             # Convert back to RGB for consistency
             denoised_uint8 = cv2.cvtColor(denoised_bgr_uint8, cv2.COLOR_BGR2RGB)
        else:
            print(f"Warning: apply_denoise received image with unexpected shape {image.shape}. Returning original.")
            # Return original image scaled back to float32
            return image.astype(np.float32)


        # Convert denoised uint8 back to float32, scaling back to original range approx.
        denoised_float = denoised_uint8.astype(np.float32) / 255.0
        if max_val > min_val:
            denoised_float = denoised_float * (max_val - min_val) + min_val

        return denoised_float.astype(np.float32) # Ensure output is float32

    except cv2.error as cv_err:
         print(f"OpenCV Error during denoising: {cv_err}")
         traceback.print_exc()
         # Return original image scaled back to float32
         return image.astype(np.float32)
    except Exception as e:
        print(f"Unexpected error during denoising: {e}")
        traceback.print_exc()
        # Return original image scaled back to float32
        return image.astype(np.float32)