"""
Module pour la détection et la correction des pixels chauds dans les images astronomiques.
"""
import numpy as np
import cv2
import traceback


# Spike-ratio isolation factor for the CFA-domain gross-isolated-defect
# detector (see ``detect_and_correct_hot_pixels_cfa``): a candidate photosite
# is corrected only if it is at least this many times brighter than its
# brightest immediate full-resolution neighbour.  A healthy compact star core
# (FWHM ~ 1.0) peaks at ~16x its orthogonal neighbour, so a 20x factor keeps
# every coherent PSF untouched while still catching gross isolated defects
# (hundreds to thousands of times above their immediate neighbourhood).
_CFA_SPIKE_FACTOR = 20.0


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


# Bayer/CFA layout vocabulary shared by the CFA-domain cosmetic correction
# and its callers (queue_manager / alignment / geometry_reference).
BAYER_PATTERNS = ("GRBG", "RGGB", "GBRG", "BGGR")


def is_bayer_pattern(pattern):
    """Return True when ``pattern`` names one of the four 2x2 Bayer CFA layouts.

    Reliable CFA recognition only: a header ``BAYERPAT`` (or the user default)
    that is one of ``GRBG`` / ``RGGB`` / ``GBRG`` / ``BGGR``.  Anything else
    (missing, ``None``, non-string, or an unrecognized value) is NOT a
    reliable CFA signal and must fall back to the existing RGB/non-CFA path.
    Recognition is value-only (the pattern string): there is NO inference from
    the array shape, and the configured ``header.get("BAYERPAT", default)``
    fallback is preserved (never silently removed).
    """
    return isinstance(pattern, str) and pattern.upper() in BAYER_PATTERNS


def detect_and_correct_hot_pixels_cfa(
    image, pattern="RGGB", threshold=3.0, neighborhood_size=5
):
    """Detect and correct defective photosites in the CFA (Bayer) domain.

    Runs BEFORE debayer on a 2D Bayer mosaic and uses SAME-COLOR neighbors
    only for the replacement value: a photosite is compared against the median
    and spread of its own color plane (the 2x2-mosaic neighbours at even
    row/column offsets), never against cross-color neighbours for replacement.
    Conservative, fully vectorized (``scipy.ndimage``), and free of any
    ZeCalibrator dependency / provenance criterion — it works identically for
    raw and already-calibrated CFA FITS.

    Gross-isolated-defect detection: a photosite is corrected only if it is
    (a) bright relative to its same-color neighbourhood AND (b) isolated — its
    immediate full-resolution neighbours (cross-phase included) are NOT bright
    (no coherent PSF / star structure).  Cross-phase neighbours answer only
    "is there coherent real spatial structure here?" and are never used for the
    replacement value.  A border band equal to the kernel radius (2 px for
    5x5) is never corrected (no edge candidates; reflect-mode border mixing is
    avoided).

    Parameters
    ----------
    image : numpy.ndarray
        2D Bayer mosaic (HxW, float or int).
    pattern : str
        One of ``GRBG`` / ``RGGB`` / ``GBRG`` / ``BGGR`` (recorded in the
        diagnostics only; the correction itself is pattern-agnostic).
    threshold : float
        Sigma multiplier above the local same-color median for a photosite
        to be considered hot (mirrors the RGB helper).
    neighborhood_size : int
        Full-mosaic odd window (>= 3); the same-color sub-window is derived
        from it (even-even offsets only).

    Returns
    -------
    (corrected_2d, diagnostics)
        ``corrected_2d`` has the same dtype/shape as the input; ``diagnostics``
        is a dict with ``enabled`` / ``pattern`` / ``candidates`` /
        ``corrected`` (``candidates == corrected``: every flagged photosite is
        replaced with its same-color median).
    """
    from scipy import ndimage as ndi

    if image is None:
        return None, {
            "enabled": False,
            "pattern": pattern,
            "candidates": 0,
            "corrected": 0,
        }

    pattern_upper = (
        pattern.upper() if isinstance(pattern, str) else str(pattern).upper()
    )
    if pattern_upper not in BAYER_PATTERNS:
        raise ValueError(
            f"Motif Bayer '{pattern}' non supporté. Options: "
            f"{', '.join(BAYER_PATTERNS)}."
        )

    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError(
            f"La correction CFA ne s'applique qu'aux images 2D. "
            f"Shape reçue: {img.shape}"
        )

    original_dtype = img.dtype
    img_float = img.astype(np.float32, copy=False)

    # Ensure the window is odd and >= 3 (mirrors the RGB helper).
    if neighborhood_size % 2 == 0:
        neighborhood_size += 1
    neighborhood_size = max(3, neighborhood_size)

    # Same-color footprint: True at even-even offsets from the window center
    # (the Bayer mosaic repeats every 2 pixels in each axis, so every pixel's
    # same-color neighbours live at even row/column offsets).
    k = int(neighborhood_size)
    center = k // 2
    radius = center  # border band = kernel radius (2 for 5x5)
    offs = np.arange(k) - center
    even = offs % 2 == 0
    footprint = even[:, None] & even[None, :]  # (k, k) bool

    # Same-color median reference (reflect border, like cv2.medianBlur).
    # Used ONLY for same-color replacement.
    median_same = ndi.median_filter(
        img_float, footprint=footprint, mode="reflect"
    )

    # Same-color neighbour mean / mean-square EXCLUDING the center photosite
    # (excluded so its own — possibly hot / star — brightness does not inflate
    # the scale).
    nc_footprint = footprint.copy()
    nc_footprint[center, center] = False
    nc_weight = nc_footprint.astype(np.float32)
    nc_kernel = nc_weight / float(nc_weight.sum())
    nc_mean = ndi.correlate(img_float, nc_kernel, mode="reflect")
    nc_mean_sq = ndi.correlate(
        img_float * img_float, nc_kernel, mode="reflect"
    )
    # Variance floor of 0.0 (never an absolute unit floor) only guards the
    # sqrt against catastrophic-cancellation negatives; the classification
    # stays invariant under positive multiplicative scaling.  No std floor:
    # a genuinely local (relative) band degenerates to ``img > median`` when
    # the same-color neighbourhood is exactly flat, which still flags a lone
    # bright photosite and never flags a flat background.
    nc_std = np.sqrt(np.maximum(nc_mean_sq - nc_mean * nc_mean, 0.0))

    # Immediate full-resolution neighbourhood MAXIMUM (the 8 adjacent
    # photosites, cross-phase included).  This answers ONLY "is there coherent
    # real spatial structure (a PSF) here?" — it is never used for the
    # replacement value.
    ring = np.ones((3, 3), dtype=bool)
    ring[1, 1] = False
    near_max = ndi.maximum_filter(
        img_float, footprint=ring, mode="reflect"
    )

    # Candidate: bright relative to its same-color neighbourhood.
    candidate = img_float > (median_same + threshold * nc_std)

    # Isolation (spike test): the candidate is a genuine isolated defect iff it
    # is a spike — at least ``_CFA_SPIKE_FACTOR`` brighter than its brightest
    # immediate full-resolution neighbour.  A healthy star core (coherent PSF)
    # has a much lower peak-to-neighbour ratio (<= ~16x for FWHM=1), so it is
    # never a spike; a defective photosite is orders of magnitude above its
    # immediate neighbourhood.  The ratio is PURELY RELATIVE (no ``max(near_max,
    # 1)`` absolute unit floor): multiplying the whole mosaic by a positive
    # scalar scales both sides identically, so the classification is invariant
    # under positive multiplicative scaling (normalized [0,1] floats and
    # 16-bit raw counts alike).
    spike = img_float > _CFA_SPIKE_FACTOR * near_max

    # Border band: skip the kernel radius (reflect-mode border mixing must
    # never produce correction candidates).
    H, W = img_float.shape
    interior = np.zeros((H, W), dtype=bool)
    interior[radius:H - radius, radius:W - radius] = True

    hot_mask = candidate & spike & interior

    corrected = img_float.copy()
    corrected[hot_mask] = median_same[hot_mask]

    n_corrected = int(np.count_nonzero(hot_mask))

    # Convert back to the original data type.
    if np.issubdtype(original_dtype, np.integer):
        min_val, max_val = (
            np.iinfo(original_dtype).min,
            np.iinfo(original_dtype).max,
        )
        corrected = np.clip(corrected, min_val, max_val).astype(original_dtype)
    else:
        corrected = corrected.astype(original_dtype)

    diagnostics = {
        "enabled": True,
        "pattern": pattern_upper,
        "candidates": n_corrected,
        "corrected": n_corrected,
    }
    return corrected, diagnostics
