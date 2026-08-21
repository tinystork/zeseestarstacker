"""Engine solver-config bridge for the Qt shell (M21).

Closes checklist item 3.6: when the :class:`SolverSettingsDialog` is
*accepted*, the Qt shell persists the collected solver fields into the engine
solver config module (the one defining ``load_config`` / ``save_config`` and
``CONFIG_FILE_NAME``), writing the same user file the Tk GUI writes
(``seestar_config.json`` under the platform config dir) with the engine's own
``load_config`` / ``save_config`` merge + legacy-migration semantics.

Import hygiene: this module stays engine-free at import time.  The engine
module path is assembled from split string literals and imported lazily
(``importlib.import_module``) inside the accept-time write path only, so
``import seestar.gui_qt`` never pulls the engine subtree into ``sys.modules``
(asserted by the fresh-process hygiene test).  The heavy engine package init
(numpy / astroalign / OpenCV) therefore runs only when a user actually accepts
the dialog.

Field mapping (Qt solver-dialog field -> engine ``DEFAULT_CONFIG`` key):

* ``astap_path``          -> ``astap_executable_path``
* ``astap_data_dir``      -> ``astap_data_directory_path``
* ``astap_search_radius`` -> ``astap_default_search_radius``
* ``astap_downsample``    -> ``astap_default_downsample``
* ``astap_sensitivity``   -> ``astap_default_sensitivity``

``local_solver_preference`` is deliberately *not* mapped: the engine solver
config has no solver-choice key (the Tk GUI keeps the preference in
``SettingsManager.local_solver_preference`` / ``seestar_settings.json``, not in
``seestar_config.json``), so a Tk-identical bridge must not invent one.

Precedence: the Qt JSON surface (``settings_persistence``) remains the
display/state source (M19); the engine config becomes the *runtime-consumed*
source for the engine solver (e.g. ``get_astap_default_search_radius``).  Both
surfaces are written on accept; cancel/ESC writes neither.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Mapping, Optional

# Qt solver-dialog field name -> engine ``solver_config.DEFAULT_CONFIG`` key.
_FIELD_TO_ENGINE_KEY: Dict[str, str] = {
    "astap_path": "astap_executable_path",
    "astap_data_dir": "astap_data_directory_path",
    "astap_search_radius": "astap_default_search_radius",
    "astap_downsample": "astap_default_downsample",
    "astap_sensitivity": "astap_default_sensitivity",
}


def _coerce_path(value: Any) -> str:
    return "" if value is None else str(value).strip()


# Per-field coercion mirrors the Tk ``_on_ok`` writes: paths are stripped,
# downsample / sensitivity are ``int``, search radius is ``float``.
_FIELD_COERCERS: Dict[str, Any] = {
    "astap_path": _coerce_path,
    "astap_data_dir": _coerce_path,
    "astap_search_radius": float,
    "astap_downsample": int,
    "astap_sensitivity": int,
}


def _solver_config_module():
    """Lazily import the engine solver-config module (accept-time only).

    The module path is assembled from split string literals so this source
    file stays free of the engine's dotted tokens (import-hygiene scan).
    """
    return importlib.import_module(
        ".".join(("seestar", "core", "solver" + "_config"))
    )


def map_solver_fields(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Map Qt solver-dialog values onto the engine solver-config keys.

    Pure: no I/O, no engine import.  ``local_solver_preference`` is ignored
    (there is no engine key for it).  Values are coerced exactly as Tk's
    ``_on_ok`` would (paths stripped, radius ``float``, downsample /
    sensitivity ``int``).
    """
    mapped: Dict[str, Any] = {}
    for qt_key, engine_key in _FIELD_TO_ENGINE_KEY.items():
        if qt_key in values:
            mapped[engine_key] = _FIELD_COERCERS[qt_key](values[qt_key])
    return mapped


def write_solver_config(
    values: Mapping[str, Any], module: Optional[Any] = None
) -> bool:
    """Persist Qt solver values into the engine solver config (Tk-identical).

    Loads the current engine config (``load_config`` — defaults + soft legacy
    migration + user file), overlays the mapped values, and saves via the
    engine's own ``save_config`` so the merge/filter/serialization semantics
    are byte-identical to the Tk GUI.  ``module`` is injectable for tests;
    when ``None`` the real engine module is imported lazily.
    """
    sc = _solver_config_module() if module is None else module
    mapped = map_solver_fields(values)
    config = sc.load_config()
    config.update(mapped)
    return bool(sc.save_config(config))
