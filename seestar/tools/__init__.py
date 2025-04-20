"""
Package tools pour Seestar - fournit des outils supplémentaires
pour le traitement des images astronomiques.
"""

from .stretch import Stretch, apply_stretch, save_fits_as_png

__all__ = ['Stretch', 'apply_stretch', 'save_fits_as_png']