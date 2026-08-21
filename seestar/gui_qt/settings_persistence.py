"""Pure-stdlib JSON persistence helper for the Qt settings state (M8).

This module is the file-backed persistence layer for
:class:`~seestar.gui_qt.settings_state.QtSettingsState`.  It deliberately knows
nothing about Qt, Tk, the scientific engine, or the settings *model* itself: it
only loads and saves a plain JSON ``dict`` and delegates type coercion /
unknown-key filtering to ``QtSettingsState.from_dict``.

Design constraints honoured here:

* **No Qt, no Tk, no numpy, no engine imports.**  Only ``json`` / ``os``, so it
  can be unit-tested in complete isolation.
* **Never crash the GUI.**  A missing file is the first-run default and a
  corrupt file must not prevent the window opening, so both load paths return an
  empty dict; a save failure returns ``False`` instead of raising.
* **Deterministic output.**  ``sort_keys`` + ``indent=2`` + ``ensure_ascii=False``
  (UTF-8) so the file is human-readable and byte-stable for tests.
* **Tk-compatible default path.**  The default is ``seestar_settings.json`` in
  the current working directory — the same convention the Tk
  ``SettingsManager`` already uses — so the non-default Qt shell never invents a
  surprising new location.  Callers (and tests) may inject any path instead.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

DEFAULT_SETTINGS_FILENAME = "seestar_settings.json"


def default_settings_path() -> str:
    """Return the default settings JSON path (CWD, matching the Tk convention)."""
    return os.path.join(os.getcwd(), DEFAULT_SETTINGS_FILENAME)


def load_settings_json(path: Optional[str]) -> Dict[str, Any]:
    """Load a settings dict from ``path``; return ``{}`` on missing/corrupt file.

    A missing file, an unreadable file, non-JSON content, or a JSON document
    that is not a mapping all degrade to an empty dict so the caller always gets
    the code defaults and the GUI always opens.  ``None``/empty ``path`` is
    treated as "persistence disabled".
    """
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def save_settings_json(path: Optional[str], data: Dict[str, Any]) -> bool:
    """Write ``data`` to ``path`` as UTF-8 JSON; return success (``bool``).

    Never raises: a write error returns ``False`` so a save failure can never
    take the GUI down.  ``sort_keys`` keeps the file deterministic for tests;
    ``ensure_ascii=False`` keeps non-ASCII paths human-readable.
    """
    if not path:
        return False
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        return True
    except OSError:
        return False
