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


def estimate_batch_size(sample_image_path=None, available_memory_percentage=70,
                        queue_length=None):
    """
    Estime la taille de lot optimale en fonction de la mémoire disponible.

    Phase B1 (canonical batch contract): this legacy entry point is now a thin
    wrapper over the independently-testable AutoBatch planner
    (``seestar.core.batch_contract.AutoBatchPlanner`` / ``plan_auto_batch``).
    The wrapper preserves the historical numeric behavior exactly
    (memory_factor=6, safety_factor=1.5, clamp [3, 50], fallback 10, ~4 MP
    fallback footprint) while the planner kernel itself is injectable,
    backend-independent and conservative on memory-discovery failure.

    Parameters:
        sample_image_path: Chemin vers une image exemple pour estimer la taille mémoire
        available_memory_percentage: Pourcentage de la mémoire disponible à utiliser (0-100)
        queue_length: Nombre connu d'échantillons (file statique).  Quand il est
            fourni, B_resolved ne dépasse jamais ce nombre d'échantillons.

    Returns:
        int: Taille de lot estimée, au moins 3 et au plus 50
    """
    from .batch_contract import AutoBatchPlanner

    # Legacy-compatible window: historical ``estimate_batch_size`` returned at
    # least 3 and at most 50 for every Auto resolution.
    min_batch = 3
    max_batch = 50
    # Default batch size if estimation fails (conservative fallback)
    default_batch_size = 10

    if not _psutil_available:
        print("psutil not available, using default batch size:", default_batch_size)
        return min(default_batch_size, max_batch)

    def _memory_query():
        # Obtenir la mémoire disponible (en octets)
        return int(psutil.virtual_memory().available)

    try:
        image_hw = None
        if sample_image_path and os.path.exists(sample_image_path):
            try:
                loaded_tuple = load_and_validate_fits(sample_image_path)
                if loaded_tuple and loaded_tuple[0] is not None:
                    img_data_for_estimation = loaded_tuple[0]
                    image_hw = tuple(int(v) for v in img_data_for_estimation.shape[:2])
                else:
                    raise ValueError(
                        f"Failed to load sample image: {sample_image_path}"
                    )
            except Exception as img_e:
                print(
                    f"Warning: Could not load/analyze sample image "
                    f"{sample_image_path} for size estimation: {img_e}"
                )
                image_hw = None  # Fallback footprint
        else:
            print("Warning: No valid sample image path provided for size estimation.")

        try:
            percentage = float(available_memory_percentage)
        except (TypeError, ValueError):
            percentage = 70.0
        fraction = max(0.0, min(100.0, percentage)) / 100.0

        # Memory discovery is delegated to the planner kernel through the
        # injectable query; a failed query returns the conservative fallback.
        planner = AutoBatchPlanner(
            memory_query=_memory_query,
            image_hw=image_hw,
            memory_factor=6,
            safety_factor=1.5,
            min_batch=min_batch,
            max_batch=max_batch,
            fallback_batch=default_batch_size,
            usable_memory_fraction=fraction,
        )
        estimated = planner.resolve(queue_length=queue_length)
        print(f"Taille de lot estimée: {estimated}")
        return estimated

    except Exception as e:
        print(f"Erreur lors de l'estimation de la taille de lot: {e}")
        traceback.print_exc()
        print(f"Utilisation de la taille de lot par défaut : {default_batch_size}")
        return min(default_batch_size, max_batch)


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


