"""PySide6 (Qt) GUI shell package — non-default.

This package contains the minimal PySide6 shell used as an architectural
foothold for the Tk → Qt parity migration.  It is kept separate from
:mod:`seestar.gui` (the Tk GUI) so the two toolkits are never mixed in a
single module.

Importing this package pulls in PySide6.  It does NOT import the Tk GUI and
does NOT touch the scientific engine.  The run-request builder
(``run_config.py``) is reached through :mod:`seestar.gui_qt.run_bridge`, which
re-exports the canonical :mod:`seestar.gui.run_config` module.  Because the
``seestar`` and ``seestar.gui`` package inits are lazy, neither the engine nor
Tk is ever pulled in.
"""

from __future__ import annotations

from .app import create_application, run_qt_app
from .main_window import (
    DEFAULT_TITLE,
    DRIZZLE_MODES,
    SOLVER_PREFERENCES,
    STACKING_MODES,
    MainWindow,
)
from .settings_state import QtSettingsState

__all__ = [
    "MainWindow",
    "DEFAULT_TITLE",
    "QtSettingsState",
    "STACKING_MODES",
    "DRIZZLE_MODES",
    "SOLVER_PREFERENCES",
    "create_application",
    "run_qt_app",
]
