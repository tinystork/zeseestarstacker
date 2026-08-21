"""Visual resources for the Qt shell (window icon + empty-preview background).

Pure display helpers with no engine / Tk / astropy dependency.  They load the
packaged PNGs (``seestar/icon/*.png``) through :mod:`importlib.resources`
(stdlib) so the shell works identically from a source checkout and from an
installed wheel (including a zip wheel), and they are *best-effort*: a missing
or undecodable resource returns ``None`` so the window still opens with no icon
and the empty preview degrades to the previous cleared pixmap.

The package-data declaration lives in ``pyproject.toml``
(``[tool.setuptools.package-data] seestar = ["icon/*.png"]``).  The ``icon``
directory is a *data* directory, not a sub-package: it has no ``__init__.py``,
which is fine because :func:`importlib.resources.files("seestar").joinpath`
traverses data files under a package without requiring the sub-directory to be
importable.
"""

from __future__ import annotations

import importlib.resources
from typing import Optional

from PySide6.QtGui import QIcon, QPixmap

_ICON_DIR = "icon"

# Decode-once cache so constructing many ``MainWindow`` instances (the test
# suite builds hundreds) does not re-decode the ~1.5 MB ``back.png`` every
# time.  The values are full-resolution pixmaps; callers scale as needed.
_cache: dict = {}


def _read_resource_bytes(name: str) -> Optional[bytes]:
    """Return the raw bytes of ``seestar/icon/<name>`` (best-effort, never raises)."""
    try:
        resource = importlib.resources.files("seestar").joinpath(_ICON_DIR, name)
        if resource.is_file():
            return resource.read_bytes()
    except Exception:
        return None
    return None


def load_window_icon() -> Optional[QIcon]:
    """Load the packaged ``icon.png`` as a :class:`QIcon` (or ``None``).

    Best-effort: a missing/undecodable icon returns ``None`` and the caller
    simply leaves the default (empty) window icon.
    """
    cached = _cache.get("icon")
    if cached is not None:
        return cached
    data = _read_resource_bytes("icon.png")
    if not data:
        return None
    pixmap = QPixmap()
    if not pixmap.loadFromData(data):
        return None
    icon = QIcon(pixmap)
    _cache["icon"] = icon
    return icon


def load_empty_preview_pixmap() -> Optional[QPixmap]:
    """Load the packaged ``back.png`` as a :class:`QPixmap` (or ``None``).

    Best-effort: a missing/undecodable background returns ``None`` and the
    caller keeps the previous cleared (null) empty-preview pixmap.
    """
    cached = _cache.get("back")
    if cached is not None:
        return cached
    data = _read_resource_bytes("back.png")
    if not data:
        return None
    pixmap = QPixmap()
    if not pixmap.loadFromData(data):
        return None
    _cache["back"] = pixmap
    return pixmap
