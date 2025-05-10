# --- START OF FILE seestar/gui/__init__.py ---
"""
Package gui pour Seestar - fournit l'interface graphique
pour le traitement des images astronomiques.
(Fichiers redondants retirés: processing.py, ui_components.py)
"""

from .main_window import SeestarStackerGUI
# Expose other components if needed directly elsewhere, but usually main_window is enough
from .preview import PreviewManager
from .histogram_widget import HistogramWidget # Added import
from .file_handling import FileHandlingManager
from .progress import ProgressManager
from .settings import SettingsManager

__all__ = [
    'SeestarStackerGUI',
    'PreviewManager',
    'HistogramWidget',      # Added export
    'FileHandlingManager',
    'ProgressManager',
    'SettingsManager',
    'ToolTip',
    ]
# --- END OF FILE seestar/gui/__init__.py ---