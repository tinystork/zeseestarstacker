"""
Seestar: Un outil d'empilement et de traitement d'images astronomiques.

Seestar est conçu pour aligner et empiler des images astronomiques afin
d'améliorer le rapport signal-bruit des observations astrophotographiques.

Import hygiene (M2): ``import seestar`` must stay cheap.  It must NOT import the
scientific engine (``seestar.core`` / ``seestar.alignment`` /
``seestar.enhancement`` / ``seestar.queuep``) nor the Tk GUI (``seestar.gui``).
Those heavy subtrees are reached lazily through :func:`__getattr__`, so
``import seestar.gui_qt`` never pulls in Tk or the engine, while
``seestar.SeestarAligner``, ``seestar.SeestarStackerGUI``, etc. keep working on
first access.
"""

from __future__ import annotations

import importlib

__version__ = "8.2.3"
__codename__ = "Phoenix consedit"  # including zenalyser and hierarchical auto stacking
__author__ = "Tinystork"

# Public name -> (module, attribute).  ``attribute=None`` means "return the
# module object itself" (used for the ``reproject_utils`` submodule re-export).
_LAZY_IMPORTS = {
    # core (scientific engine) — imported only on first access
    "SeestarAligner": ("seestar.core", "SeestarAligner"),
    "load_and_validate_fits": ("seestar.core", "load_and_validate_fits"),
    "debayer_image": ("seestar.core", "debayer_image"),
    "detect_and_correct_hot_pixels": ("seestar.core", "detect_and_correct_hot_pixels"),
    "save_fits_image": ("seestar.core", "save_fits_image"),
    "save_preview_image": ("seestar.core", "save_preview_image"),
    "estimate_batch_size": ("seestar.core", "estimate_batch_size"),
    "apply_denoise": ("seestar.core", "apply_denoise"),
    "collect_headers": ("seestar.core", "collect_headers"),
    "compute_final_output_grid": ("seestar.core", "compute_final_output_grid"),
    # tools
    "StretchPresets": ("seestar.tools", "StretchPresets"),
    "ColorCorrection": ("seestar.tools", "ColorCorrection"),
    "apply_auto_stretch": ("seestar.tools", "apply_auto_stretch"),
    "apply_auto_white_balance": ("seestar.tools", "apply_auto_white_balance"),
    "apply_enhanced_stretch": ("seestar.tools", "apply_enhanced_stretch"),
    "save_fits_as_png": ("seestar.tools", "save_fits_as_png"),
    # enhancement submodule (re-exported as a top-level name)
    "reproject_utils": ("seestar.enhancement.reproject_utils", None),
    # GUI (default Tk entry point)
    "SeestarStackerGUI": ("seestar.gui", "SeestarStackerGUI"),
}

__all__ = [
    "__version__",
    "__author__",
    *sorted(_LAZY_IMPORTS),
]


def __getattr__(name):
    entry = _LAZY_IMPORTS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = entry
    module = importlib.import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():  # pragma: no cover - helper for interactive use
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
