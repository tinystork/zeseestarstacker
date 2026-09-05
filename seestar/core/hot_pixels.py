"""
Module pour la détection et la correction des pixels chauds dans les images astronomiques.
"""
import numpy as np
import cv2
import traceback


def detect_and_correct_hot_pixels(image, threshold=3.0, neighborhood_size=5):
    """
    Détecte et corrige les pixels chauds dans une image.

    Parameters:
        image (numpy.ndarray): Image à traiter (HxW ou HxWx3, float ou int)
        threshold (float): Seuil en écarts-types pour considérer un pixel comme "chaud"
        neighborhood_size (int): Taille du voisinage pour le calcul de la médiane (doit être impair)

    Returns:
        numpy.ndarray: Image avec pixels chauds corrigés (même dtype que l'entrée)
    """
    if image is None:
        print("Warning: detect_and_correct_hot_pixels received None image.")
        return None

    # Ensure neighborhood size is odd and >= 3
    if neighborhood_size % 2 == 0:
        # print(f"Warning: neighborhood_size was even ({neighborhood_size}), adjusting to {neighborhood_size + 1}.")
        neighborhood_size += 1
    neighborhood_size = max(3, neighborhood_size)
    ksize = (neighborhood_size, neighborhood_size) # Kernel size tuple

    original_dtype = image.dtype
    # Work with float32 for calculations
    img_float = image.astype(np.float32, copy=True) # Work on a copy
    corrected_float = img_float # Modify in place

    is_color = img_float.ndim == 3 and img_float.shape[-1] == 3
    std_dev = None # Initialize std_dev

    try:
        if is_color:
            # Process each channel separately
            for c in range(img_float.shape[2]):
                channel = corrected_float[:, :, c] # Work directly on the copy

                # --- Median Filter (CPU) ---
                # Use median filter for both detection reference and replacement value
                median_filtered_channel = cv2.medianBlur(channel, neighborhood_size)

                # --- Mean and Std Dev Calculation (CPU) ---
                mean = cv2.blur(channel, ksize)
                mean_sq = cv2.blur(channel**2, ksize)

                # Calculate standard deviation
                std_dev = np.sqrt(np.maximum(mean_sq - mean**2, 1e-10)) # Add epsilon

                # Prevent near-zero standard deviation issues
                std_dev_floor = 1e-5
                if np.issubdtype(original_dtype, np.integer): std_dev_floor = 1.0
                std_dev = np.maximum(std_dev, std_dev_floor)

                # Identify hot pixels: significantly brighter than the local *median*
                hot_pixels_mask = channel > (median_filtered_channel + threshold * std_dev)

                # Replace hot pixels with the median value of their neighborhood
                channel[hot_pixels_mask] = median_filtered_channel[hot_pixels_mask]

        else: # Grayscale image
            channel = corrected_float # Work directly on the copy

            # --- Median Filter (CPU) ---
            median_filtered_channel = cv2.medianBlur(channel, neighborhood_size)

            # --- Mean and Std Dev Calculation (CPU) ---
            mean = cv2.blur(channel, ksize)
            mean_sq = cv2.blur(channel**2, ksize)

            # Calculate standard deviation
            std_dev = np.sqrt(np.maximum(mean_sq - mean**2, 1e-10))

            std_dev_floor = 1e-5
            if np.issubdtype(original_dtype, np.integer): std_dev_floor = 1.0
            std_dev = np.maximum(std_dev, std_dev_floor)

            # Identify and correct hot pixels
            hot_pixels_mask = channel > (median_filtered_channel + threshold * std_dev)
            channel[hot_pixels_mask] = median_filtered_channel[hot_pixels_mask]

        # Convert back to the original data type
        if np.issubdtype(original_dtype, np.integer):
             min_val, max_val = np.iinfo(original_dtype).min, np.iinfo(original_dtype).max
             corrected_img = np.clip(corrected_float, min_val, max_val).astype(original_dtype)
        else: # Float types
             corrected_img = corrected_float.astype(original_dtype)

        return corrected_img

    except Exception as e:
        print(f"Erreur dans detect_and_correct_hot_pixels: {e}")
        traceback.print_exc()
        # Return the original image in case of unexpected errors
        return image
# --- END OF MODIFIED seestar/core/hot_pixels.py --
