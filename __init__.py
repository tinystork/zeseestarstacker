"""
Seestar: Un outil d'empilement et de traitement d'images astronomiques.

Seestar est conçu pour aligner et empiler des images astronomiques afin
d'améliorer le rapport signal-bruit des observations astrophotographiques.
"""

__version__ = "1.1.0" # Incremented version - Simplified
__author__ = "Seestar Team"

# Core functionalities
from seestar.core import (
    SeestarAligner,
    load_and_validate_fits,
    debayer_image,
    detect_and_correct_hot_pixels,
    save_fits_image,
    save_preview_image,
    estimate_batch_size,
    apply_denoise
)

# Tools
from seestar.tools import (
    Stretch,
    apply_stretch,
    save_fits_as_png
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
    'Stretch',
    'apply_stretch',
    'save_fits_as_png',
    # GUI
    'SeestarStackerGUI',
    # Package Info
    '__version__',
    '__author__'
]