"""
Seestar: Un outil d'empilement et de traitement d'images astronomiques.

Seestar est conçu pour aligner et empiler des images astronomiques afin
d'améliorer le rapport signal-bruit des observations astrophotographiques.
"""

__version__ = "2.9.0" # Version incrémentée - zemosaic intégré
__author__ = "Tinystork"

# Core functionalities (unchanged from your original structure)
from seestar.core import (
    SeestarAligner,
    load_and_validate_fits,
    debayer_image,
    detect_and_correct_hot_pixels,
    save_fits_image,
    save_preview_image,
    estimate_batch_size,
    apply_denoise # Keep apply_denoise function available even if GUI option removed
)

# Tools (updated imports based on the new stretch.py)
from seestar.tools import (
    StretchPresets, # Changed from Stretch class
    ColorCorrection, # Added
    apply_auto_stretch, # Added helper
    apply_auto_white_balance, # Added helper
    apply_enhanced_stretch, # Kept this name, assuming it's useful elsewhere
    save_fits_as_png # Kept
)

# GUI (expose the main class)
from seestar.gui import SeestarStackerGUI

__all__ = [
    # Core
    'SeestarAligner',
    'load_and_validate_fits',
    'debayer_image',
    'detect_and_correct_hot_pixels',
    'save_fits_image',
    'save_preview_image',
    'estimate_batch_size',
    'apply_denoise',
    # Tools
    'StretchPresets',
    'ColorCorrection',
    'apply_auto_stretch',
    'apply_auto_white_balance',
    'apply_enhanced_stretch',
    'save_fits_as_png',
    # GUI
    'SeestarStackerGUI',
    # Package Info
    '__version__',
    '__author__'
]
# --- END OF FILE seestar/__init__.py ---