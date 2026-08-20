"""Non-default PySide6 launch helper.

Run explicitly with::

    python -m seestar.qt_main

This does NOT change the default ``zeseestarstacker = seestar.main:main``
entry point; it exists so the experimental Qt shell can be launched during the
migration without disturbing the Tk application.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .gui_qt.app import run_qt_app


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_qt_app(argv)


if __name__ == "__main__":
    raise SystemExit(main())
