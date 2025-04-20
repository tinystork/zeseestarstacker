"""
Package core pour Seestar - fournit les fonctionnalités de base pour le traitement des images astronomiques.
"""

from .image_processing import (
    load_and_validate_fits,
    debayer_image,
    save_fits_image,
    save_preview_image,
    # load_and_validate_fits included above covers get_header option
)

from .hot_pixels import detect_and_correct_hot_pixels
from .utils import estimate_batch_size, apply_denoise
from .alignment import SeestarAligner
# Removed: from .stacking import ProgressiveStacker

__all__ = [
    'load_and_validate_fits',
    'debayer_image',
    'detect_and_correct_hot_pixels',
    'save_fits_image',
    'save_preview_image',
    'estimate_batch_size',
    'apply_denoise',
    'SeestarAligner',
    # 'ProgressiveStacker' # Removed
]