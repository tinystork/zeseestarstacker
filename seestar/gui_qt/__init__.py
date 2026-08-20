"""PySide6 (Qt) GUI shell package — non-default.

This package contains the minimal PySide6 shell used as an architectural
foothold for the Tk → Qt parity migration.  It is kept separate from
:mod:`seestar.gui` (the Tk GUI) so the two toolkits are never mixed in a
single module.

Importing this package pulls in PySide6.  It does NOT import the Tk GUI and
does NOT touch the scientific engine.
"""

from __future__ import annotations

from .app import create_application, run_qt_app
from .main_window import DEFAULT_TITLE, MainWindow

__all__ = [
    "MainWindow",
    "DEFAULT_TITLE",
    "create_application",
    "run_qt_app",
]
