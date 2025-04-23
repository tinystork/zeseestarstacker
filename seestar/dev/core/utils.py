# --- START OF FILE seestar/core/utils.py ---
"""
Fonctions utilitaires pour le traitement d'images astronomiques.
"""
import numpy as np
import cv2
import os # Added for exists check
import traceback # Added for better error reporting
from .image_processing import load_and_validate_fits # Keep relative import

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

        # Estimer la taille d'une image en mémoire pendant le traitement
        single_image_size_bytes = 0
        if sample_image_path and os.path.exists(sample_image_path):
            try:
                # Load image to get dimensions and type (returns float32 0-1)
                img = load_and_validate_fits(sample_image_path)
                if img is None: raise ValueError("Failed to load sample image for size estimation.")

                # Estimate memory usage during processing (alignment + stacking buffer)
                # This is a rough estimate and depends heavily on the exact workflow.
                # Assume output will be color (3 channels float32) for sizing.
                # Factors to consider:
                # - Loaded image (float32, maybe 1 or 3 channels)
                # - Debayered/Preprocessed image (float32, 3 channels)
                # - Aligned image buffer (float32, 3 channels)
                # - Stacking accumulator (float32 or float64, 3 channels)
                # - Intermediate arrays in astroalign etc.
                # Let's use a factor of ~5-7 times the *output* data size (HxWx3 float32).
                memory_factor = 6
                h, w = img.shape[:2]
                channels_out = 3 # Assume color output for worst-case size
                bytes_per_float = 4
                single_image_size_bytes = h * w * channels_out * bytes_per_float * memory_factor

            except Exception as img_e:
                 print(f"Warning: Could not load/analyze sample image {sample_image_path} for size estimation: {img_e}")
                 single_image_size_bytes = 0 # Fallback
        else:
            print("Warning: No valid sample image path provided for size estimation.")
            single_image_size_bytes = 0 # Fallback


        # Fallback estimation if image loading failed or no path provided
        if single_image_size_bytes <= 0:
            # Conservative estimate for a ~4MP color image (float32) + overhead
            # 2000 * 2000 * 3 channels * 4 bytes/float * 6x overhead = ~288 MB
            print("Using fallback image size estimation (approx. 4MP color image).")
            single_image_size_bytes = 2000 * 2000 * 3 * 4 * 6


        # Safety factor for other system usage and Python overhead
        safety_factor = 1.5 # Includes buffer for OS, other apps, Python interpreter overhead

        if single_image_size_bytes <= 0: # Should not happen, but safeguard
             print("Error: Calculated image size is zero or negative.")
             return default_batch_size

        # Calculate how many images can fit in memory
        estimated_batch = int(usable_memory / (single_image_size_bytes * safety_factor))

        # Reasonable limits for batch size (at least 3 for sigma clipping, max 50 to avoid huge single batches)
        estimated_batch = max(3, min(50, estimated_batch))

        print(f"Mémoire disponible: {available_memory / (1024**3):.2f} Go "
              f"(Utilisable: {usable_memory / (1024**3):.2f} Go)")
        print(f"Taille estimée par image (avec overhead): {single_image_size_bytes / (1024**2):.2f} Mo")
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
    Input image is expected to be float32 (0-1 range).

    Parameters:
        image (numpy.ndarray): Image à débruiter (HxW ou HxWx3, float32, 0-1).
        strength (int): Force du débruitage (paramètre 'h' de fastNlMeansDenoising*).

    Returns:
        numpy.ndarray: Image débruitée (float32, 0-1).
    """
    if image is None:
        print("Warning: apply_denoise received a None image.")
        return None

    # Convert image float (0-1) to uint8 (0-255) for OpenCV
    # Use np.clip for safety before conversion
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

    try:
        denoised_uint8 = None
        h_param = float(strength) # Parameter 'h' in OpenCV functions

        if image_uint8.ndim == 2:
            # Image monochrome
            # print(f"Applying fastNlMeansDenoising with h={h_param}")
            denoised_uint8 = cv2.fastNlMeansDenoising(
                image_uint8, None, h=h_param, templateWindowSize=7, searchWindowSize=21)

        elif image_uint8.ndim == 3 and image_uint8.shape[-1] == 3:
             # Image couleur RGB (input is uint8 HxWx3)
             # fastNlMeansDenoisingColored expects BGR uint8.
             # print(f"Applying fastNlMeansDenoisingColored with h={h_param}, hColor={h_param}")
             image_bgr_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
             denoised_bgr_uint8 = cv2.fastNlMeansDenoisingColored(
                 image_bgr_uint8, None, h=h_param, hColor=h_param, # Use same strength for luma and chroma
                 templateWindowSize=7, searchWindowSize=21)
             # Convert back to RGB for consistency
             denoised_uint8 = cv2.cvtColor(denoised_bgr_uint8, cv2.COLOR_BGR2RGB)
        else:
            print(f"Warning: apply_denoise received image with unexpected shape {image.shape}. Returning original.")
            return image # Return original float32 image


        # Convert denoised uint8 back to float32 (0-1 range)
        denoised_float = denoised_uint8.astype(np.float32) / 255.0
        return denoised_float

    except cv2.error as cv_err:
         print(f"OpenCV Error during denoising: {cv_err}")
         traceback.print_exc()
         return image # Return original float32 image
    except Exception as e:
        print(f"Unexpected error during denoising: {e}")
        traceback.print_exc()
        return image # Return original float32 image
# --- END OF FILE seestar/core/utils.py ---