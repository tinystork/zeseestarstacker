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
# --- Add a global check for CUDA availability ONCE ---
_cuda_available = False
_cuda_checked = False

def check_cuda():
    """Checks if OpenCV reports CUDA devices and sets a global flag."""
    global _cuda_available, _cuda_checked
    if _cuda_checked:
        return _cuda_available
    try:
        # Make sure opencv-contrib-python is potentially installed
        if not hasattr(cv2, 'cuda'):
             print("DEBUG: cv2.cuda module not found (likely opencv-python, not opencv-contrib-python or CUDA not supported in build).")
             _cuda_available = False
        elif cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("DEBUG: CUDA device(s) detected by OpenCV.")
            cv2.cuda.printCudaDeviceInfo(cv2.cuda.getDevice()) # Print info about the default device
            _cuda_available = True
        else:
            print("DEBUG: No CUDA devices detected by OpenCV.")
            _cuda_available = False
    except Exception as e:
        print(f"DEBUG: Error checking for CUDA devices: {e}")
        _cuda_available = False
    finally:
        _cuda_checked = True
    return _cuda_available

def estimate_batch_size(sample_image_path=None, available_memory_percentage=70):
    """
    Estime la taille de lot optimale en fonction de la mémoire disponible.
    CORRIGÉ: Gère correctement le tuple retourné par load_and_validate_fits.

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
            img_data_for_estimation = None # Initialiser
            try:
                # Load image to get dimensions and type (returns float32 0-1)
                loaded_tuple = load_and_validate_fits(sample_image_path) # APPEL MODIFIÉ

                # --- DÉBUT DE LA CORRECTION ---
                if loaded_tuple and loaded_tuple[0] is not None:
                    img_data_for_estimation = loaded_tuple[0] # Déballer l'array image
                else:
                    # Si load_and_validate_fits retourne None ou si les données sont None,
                    # img_data_for_estimation restera None.
                    # Le ValueError sera levé plus bas si img_data_for_estimation est None.
                    pass # img_data_for_estimation est déjà None
                # --- FIN DE LA CORRECTION ---

                if img_data_for_estimation is None: # Vérifier après la tentative de déballage
                    raise ValueError(f"Failed to load sample image: {sample_image_path}")

                # Estimate memory usage during processing (alignment + stacking buffer)
                memory_factor = 6
                h, w = img_data_for_estimation.shape[:2] # Utiliser img_data_for_estimation
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
            print("Using fallback image size estimation (approx. 4MP color image).")
            single_image_size_bytes = 2000 * 2000 * 3 * 4 * 6


        # Safety factor for other system usage and Python overhead
        safety_factor = 1.5

        if single_image_size_bytes <= 0:
             print("Error: Calculated image size is zero or negative.")
             return default_batch_size

        estimated_batch = int(usable_memory / (single_image_size_bytes * safety_factor))
        estimated_batch = max(3, min(50, estimated_batch)) # Limites raisonnables

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


def downsample_image(image: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample an image by an integer factor using OpenCV."""

    if image is None or factor <= 1:
        return image

    try:
        h, w = image.shape[:2]
        new_w, new_h = w // factor, h // factor
        if new_w < 1 or new_h < 1:
            return image

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    except Exception:
        print("Warning: downsample_image failed; returning original image")
        traceback.print_exc(limit=1)
        return image


