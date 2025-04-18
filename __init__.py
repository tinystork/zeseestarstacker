"""
Seestar: Un outil d'empilement et de traitement d'images astronomiques.

Seestar est conçu pour aligner et empiler des images astronomiques afin
d'améliorer le rapport signal-bruit des observations astrophotographiques.
"""

__version__ = "1.0.0"
__author__ = "Seestar Team"

from seestar.core import (
    SeestarAligner, 
    ProgressiveStacker, 
    load_and_validate_fits,
    debayer_image,
    detect_and_correct_hot_pixels,
    save_fits_image,
    save_preview_image,
    estimate_batch_size
)

from seestar.queuep import (
    SeestarQueuedStacker,
    ImageDatabase,
    ImageInfo
)

from seestar.tools import (
    Stretch,
    apply_stretch,
    save_fits_as_png
)

__all__ = [
    'SeestarAligner',
    'ProgressiveStacker',
    'SeestarQueuedStacker',
    'ImageDatabase',
    'ImageInfo',
    'load_and_validate_fits',
    'debayer_image',
    'detect_and_correct_hot_pixels',
    'save_fits_image',
    'save_preview_image',
    'estimate_batch_size',
    'Stretch',
    'apply_stretch',
    'save_fits_as_png',
    '__version__',
    '__author__'
]