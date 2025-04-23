# --- START OF FILE seestar/core/hot_pixels.py ---
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

    # Ensure neighborhood size is odd
    if neighborhood_size % 2 == 0:
        print(f"Warning: neighborhood_size was even ({neighborhood_size}), adjusting to {neighborhood_size + 1}.")
        neighborhood_size += 1
    # Ensure minimum size
    neighborhood_size = max(3, neighborhood_size)


    original_dtype = image.dtype
    # Work with float32 for calculations
    # Important: Check if input could be float64 and handle appropriately if needed
    img_float = image.astype(np.float32, copy=True) # Work on a copy
    corrected_float = img_float # Modify in place

    is_color = img_float.ndim == 3 and img_float.shape[-1] == 3

    try:
        if is_color:
            # Process each channel separately
            for c in range(img_float.shape[2]):
                channel = corrected_float[:, :, c] # Work directly on the copy

                # Calculate local median (more robust to outliers than mean for hot pixels)
                # Use median filter for both detection reference and replacement value
                median_filtered_channel = cv2.medianBlur(channel, neighborhood_size)

                # Estimate local standard deviation using blur for performance
                mean = cv2.blur(channel, (neighborhood_size, neighborhood_size))
                # Use abs(channel - mean) instead of (channel-mean)**2 before blur for std dev approx
                # This is sometimes called pseudo-standard deviation, faster but less accurate
                # abs_diff_blur = cv2.blur(np.abs(channel - mean), (neighborhood_size, neighborhood_size))
                # std_dev = abs_diff_blur * np.sqrt(np.pi / 2.0) # Approximation factor

                # Sticking with more standard calculation:
                mean_sq = cv2.blur(channel**2, (neighborhood_size, neighborhood_size))
                std_dev = np.sqrt(np.maximum(mean_sq - mean**2, 1e-10)) # Add epsilon before sqrt

                # Prevent near-zero standard deviation issues
                # std_dev = np.maximum(std_dev, 1e-5 * np.nanmax(channel)) # Scale floor by max value?
                # Or simpler floor:
                std_dev_floor = 1e-5 # Adjust if needed based on data range (before normalization)
                if np.issubdtype(original_dtype, np.integer): std_dev_floor = 1.0 # Use 1 for integers
                std_dev = np.maximum(std_dev, std_dev_floor)


                # Identify hot pixels: significantly brighter than the local *median*
                # Using median as reference is better for hot pixels than using mean
                hot_pixels_mask = channel > (median_filtered_channel + threshold * std_dev)

                # Replace hot pixels with the median value of their neighborhood
                channel[hot_pixels_mask] = median_filtered_channel[hot_pixels_mask]

        else: # Grayscale image
            channel = corrected_float # Work directly on the copy
            median_filtered_channel = cv2.medianBlur(channel, neighborhood_size)

            mean = cv2.blur(channel, (neighborhood_size, neighborhood_size))
            mean_sq = cv2.blur(channel**2, (neighborhood_size, neighborhood_size))
            std_dev = np.sqrt(np.maximum(mean_sq - mean**2, 1e-10))

            std_dev_floor = 1e-5
            if np.issubdtype(original_dtype, np.integer): std_dev_floor = 1.0
            std_dev = np.maximum(std_dev, std_dev_floor)

            hot_pixels_mask = channel > (median_filtered_channel + threshold * std_dev)
            channel[hot_pixels_mask] = median_filtered_channel[hot_pixels_mask]

        # Convert back to the original data type
        # Clip values if converting back to integer types to avoid wrap-around
        if np.issubdtype(original_dtype, np.integer):
             min_val, max_val = np.iinfo(original_dtype).min, np.iinfo(original_dtype).max
             corrected_img = np.clip(corrected_float, min_val, max_val).astype(original_dtype)
        else: # Float types
             # Ensure output range matches input if float (e.g., 0-1 if input was 0-1)
             # Assuming input float was already normalized or handled elsewhere.
             corrected_img = corrected_float.astype(original_dtype)

        return corrected_img

    except Exception as e:
        print(f"Erreur dans detect_and_correct_hot_pixels: {e}")
        traceback.print_exc()
        # Return the original image in case of unexpected errors
        return image
# --- END OF FILE seestar/core/hot_pixels.py ---