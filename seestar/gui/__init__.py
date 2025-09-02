"""Lightweight GUI package init avoiding heavy backend imports for tests.

This module provides lazy accessors for the heavy GUI classes so that
importing :mod:`seestar.gui` in environments without the full GUI stack
remains possible.  Consumers can still do ``from seestar.gui import
SeestarStackerGUI`` and the actual backend will only be imported on first
access.
"""

from importlib import import_module

from . import boring_stack  # Re-export for convenience

_LAZY_IMPORTS = {
    "SeestarStackerGUI": "main_window",
    "PreviewManager": "preview",
    "HistogramWidget": "histogram_widget",
    "FileHandlingManager": "file_handling",
    "ProgressManager": "progress",
    "SettingsManager": "settings",
}

__all__ = ["boring_stack", *sorted(_LAZY_IMPORTS)]


def __getattr__(name):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name:
        module = import_module(f".{module_name}", __name__)
        attr = getattr(module, name)
        globals()[name] = attr  # Cache for subsequent lookups
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # pragma: no cover - helper for interactive use
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))
