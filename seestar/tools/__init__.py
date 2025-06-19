"""
Package tools pour Seestar - fournit des outils supplémentaires
pour le traitement et la visualisation des images astronomiques.
"""

from .stretch import (
     StretchPresets,
     ColorCorrection,
     apply_auto_stretch,
     apply_auto_white_balance,
     apply_enhanced_stretch,  # Keep if needed
     save_fits_as_png,
)
from .file_ops import move_to_stacked

# Optionally import visu if you want it accessible via seestar.tools.visu (though it's standalone)
# from . import visu
# from . import testimg

__all__ = [
     # Stretch and Color Correction Tools
     'StretchPresets',
     'ColorCorrection',
     'apply_auto_stretch',
     'apply_auto_white_balance',
     'apply_enhanced_stretch',
     'save_fits_as_png',
     'move_to_stacked'
     # Add 'visu', 'testimg' here if you import and want to expose them
     ]
# --- END OF FILE seestar/tools/__init__.py ---
