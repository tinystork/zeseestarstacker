"""Lightweight GUI package init avoiding heavy backend imports for tests.

This module provides lazy accessors for the heavy GUI classes and the
``boring_stack`` module so that importing :mod:`seestar.gui` — including via
``seestar.gui.run_config`` from the Qt shell — does NOT pull in Tk or the
scientific engine.  Consumers can still do ``from seestar.gui import
SeestarStackerGUI`` (or ``boring_stack``) and the actual backend is only
imported on first access.
"""

from importlib import import_module

# Submodules re-exported as *module objects* (rather than an attribute of them).
_MODULE_REEXPORTS = {
    "boring_stack": "boring_stack",
}

_LAZY_IMPORTS = {
    "SeestarStackerGUI": "main_window",
    "PreviewManager": "preview",
    "HistogramWidget": "histogram_widget",
    "FileHandlingManager": "file_handling",
    "ProgressManager": "progress",
    "SettingsManager": "settings",
}

__all__ = [*sorted(_MODULE_REEXPORTS), *sorted(_LAZY_IMPORTS)]


def __getattr__(name):
    module_name = _MODULE_REEXPORTS.get(name)
    if module_name is not None:
        module = import_module(f".{module_name}", __name__)
        globals()[name] = module
        return module
    module_name = _LAZY_IMPORTS.get(name)
    if module_name:
        module = import_module(f".{module_name}", __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # pragma: no cover - helper for interactive use
    return sorted(set(globals()) | set(_LAZY_IMPORTS) | set(_MODULE_REEXPORTS))
