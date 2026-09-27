"""Lazy calibration service layer (public calibration boundary, engine-free at import).

The pure-Qt Calibration tab needs two things from the optional ZeCalibrator
integration: the provider probe (availability + status) and a session over the
chosen master folder (admission-only master counts + diagnostics).  Those live
in :mod:`seestar.calibration` (which may import ``zecalibrator``); ``gui_qt``
must never import that subtree at package-import time.

This module is the *lazy* service that reaches them on first call, using the
same import-hygiene pattern as :mod:`seestar.gui_qt.solver_service`: the module
path is assembled from split string literals so this source stays free of the
calibration engine's dotted tokens, and a fresh ``import seestar.gui_qt`` never
pulls ``zecalibrator`` (or NumPy/Astropy) into ``sys.modules`` via this path.
"""

from __future__ import annotations

import importlib


def _adapter_module():
    """Import the calibration adapter lazily (first call only)."""
    return importlib.import_module(
        ".".join(("seestar", "calibration", "zecalibrator" + "_adapter"))
    )


def check_calibration_availability():
    """Return the calibration provider probe (never raises for absent provider).

    Delegates to the adapter's defensive :func:`probe`, which returns a neutral
    ``ProviderInfo`` (AVAILABLE / NOT_INSTALLED / UNHEALTHY / INCOMPATIBLE)
    instead of raising.
    """
    return _adapter_module().probe()


def calibration_tab_should_exist(info) -> bool:
    """Return whether the Calibration tab should be shown for a probe result.

    Pure decision re-exported from the adapter (no Qt, no zecalibrator import).
    """
    return _adapter_module().calibration_tab_should_exist(info)


def open_calibration_session(root, *, cancel=None):
    """Open a calibration session for master-count/diagnostic display.

    Admission-only (role identification): no matching, no calibration, no
    science. Returns a neutral ``SessionResult``; an unavailable provider yields
    ``state == UNAVAILABLE`` (never raises).
    """
    return _adapter_module().ZeCalibratorProvider().open_session(root, cancel=cancel)


__all__ = [
    "calibration_tab_should_exist",
    "check_calibration_availability",
    "open_calibration_session",
]
