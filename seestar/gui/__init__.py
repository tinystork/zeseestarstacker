"""
Package gui pour Seestar - fournit l'interface graphique
pour le traitement des images astronomiques.
(Fichiers redondants retirés: processing.py, ui_components.py)
"""

# Delay heavy GUI imports so headless environments can import lightweight
# helpers (e.g. ``boring_stack``) without requiring a display backend.

__all__ = [
    "SeestarStackerGUI",
    "PreviewManager",
    "HistogramWidget",
    "FileHandlingManager",
    "ProgressManager",
    "SettingsManager",
    "boring_stack",
]


def __getattr__(name):  # pragma: no cover - simple lazy imports
    if name == "SeestarStackerGUI":
        from .main_window import SeestarStackerGUI
        return SeestarStackerGUI
    if name == "PreviewManager":
        from .preview import PreviewManager
        return PreviewManager
    if name == "HistogramWidget":
        from .histogram_widget import HistogramWidget
        return HistogramWidget
    if name == "FileHandlingManager":
        from .file_handling import FileHandlingManager
        return FileHandlingManager
    if name == "ProgressManager":
        from .progress import ProgressManager
        return ProgressManager
    if name == "SettingsManager":
        from .settings import SettingsManager
        return SettingsManager
    if name == "boring_stack":
        from . import boring_stack
        return boring_stack
    raise AttributeError(name)
# --- END OF FILE seestar/gui/__init__.py ---
