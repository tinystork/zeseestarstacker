"""
Seestar: Un outil d'empilement et de traitement d'images astronomiques.

Seestar est conçu pour aligner et empiler des images astronomiques afin
d'améliorer le rapport signal-bruit des observations astrophotographiques.
"""

__version__ = "6.5.0 Boring"  # including zenalyser and hierarchical auto satcking 
__author__ = "Tinystork"

import sys

# Core functionalities (unchanged from your original structure)
try:
    from seestar.core import (
        SeestarAligner,
        load_and_validate_fits,
        debayer_image,
        detect_and_correct_hot_pixels,
        save_fits_image,
        save_preview_image,
        estimate_batch_size,
        apply_denoise,  # Keep apply_denoise available if GUI option removed
        collect_headers,
        compute_final_output_grid,
    )
    _CORE_AVAILABLE = True
except Exception as e:  # pragma: no cover - optional dependency may be missing
    import logging

    logging.getLogger(__name__).warning(
        "Seestar core not available (%s). Some functionality is disabled.", e
    )
    _CORE_AVAILABLE = False

# Tools (updated imports based on the new stretch.py)
from seestar.tools import (
    StretchPresets,  # Changed from Stretch class
    ColorCorrection,  # Added
    apply_auto_stretch,  # Added helper
    apply_auto_white_balance,  # Added helper
    apply_enhanced_stretch,  # Kept this name, assuming it's useful elsewhere
    save_fits_as_png  # Kept
)

from seestar.enhancement import reproject_utils

# GUI (optional)
try:
    if _CORE_AVAILABLE:
        from seestar.gui import SeestarStackerGUI
        _GUI_AVAILABLE = True
    else:
        raise RuntimeError("Core modules missing")
except BaseException as e:  # pragma: no cover - GUI might not be present
    import logging

    sys.modules.pop("seestar.gui", None)
    logging.getLogger(__name__).warning(
        "Seestar GUI not available (%s). Running in headless mode.", e
    )
    SeestarStackerGUI = None
    _GUI_AVAILABLE = False

__all__ = [
    # Tools
    'StretchPresets',
    'ColorCorrection',
    'apply_auto_stretch',
    'apply_auto_white_balance',
    'apply_enhanced_stretch',
    'save_fits_as_png',
    'reproject_utils',
    # Package Info
    '__version__',
    '__author__'
]

if _CORE_AVAILABLE:
    __all__[0:0] = [
        'SeestarAligner',
        'load_and_validate_fits',
        'debayer_image',
        'detect_and_correct_hot_pixels',
        'save_fits_image',
        'save_preview_image',
        'estimate_batch_size',
        'apply_denoise',
        'collect_headers',
        'compute_final_output_grid',
    ]

if _GUI_AVAILABLE:
    __all__.append('SeestarStackerGUI')
# --- END OF FILE seestar/__init__.py ---
