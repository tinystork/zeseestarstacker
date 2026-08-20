"""Qt-side preflight validation for real-backend starts (M6 seam).

Before the PySide6 shell hands a run to the *real*
:class:`~seestar.gui_qt.backend_runner.SeestarQueuedStackerBackend`, it should
guard against obviously bad starts (empty input/output folders, malformed batch
or drizzle values) and fail *early* with human-readable feedback — without ever
importing the Tk ``SettingsManager`` or the scientific engine.

This module is that seam.  It is pure stdlib (no Qt, no Tk, no engine, no
numpy) so it can be imported and unit-tested in complete isolation, and it
validates a plain settings object (a :class:`QtSettingsState`, or the canonical
:class:`~seestar.gui_qt.run_bridge.RunRequest`) against a target backend mode.

Policy (kept deliberately small):

* ``backend_mode == "simulated"`` is **always permissive**: the simulated
  backend never reads the engine, so an empty/malformed settings model is fine
  for the offscreen smoke path and returns no errors.
* ``backend_mode == "seestar"`` applies the real-backend preflight: empty
  ``input_folder``/``output_folder`` are rejected, ``batch_size`` must be
  integer-like and ``>= -1`` (``-1`` = auto), and ``drizzle_group_size`` must be
  ``> 0``.
* The ``backend_factory`` injection path is **not** blocked by this preflight
  unless ``backend_mode == "seestar"`` (so fake-backend tests stay easy).  The
  preflight is keyed solely on ``backend_mode``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List

# Backend modes understood by the preflight.  Anything not ``seestar`` is
# treated as permissive (the only in-tree non-seestar mode is ``simulated``).
SEESTAR_MODE = "seestar"


def _read(settings: Any, state_attr: str, runrequest_key: str, default: Any) -> Any:
    """Read one preflight field from either a state or a RunRequest.

    A :class:`~seestar.gui_qt.run_bridge.RunRequest` exposes its values through
    ``backend_kwargs`` (a mapping with backend key names such as ``input_dir``);
    a :class:`~seestar.gui_qt.settings_state.QtSettingsState` exposes plain
    attributes (``input_folder``).  This duck-typed reader supports both without
    importing either class.
    """
    backend_kwargs = getattr(settings, "backend_kwargs", None)
    if isinstance(backend_kwargs, Mapping):
        return backend_kwargs.get(runrequest_key, default)
    return getattr(settings, state_attr, default)


def _nonempty_str(value: Any) -> bool:
    """True when ``value`` is a non-empty string after stripping whitespace."""
    return bool(str(value or "").strip())


def _as_int(value: Any) -> Any:
    """Return ``int(value)`` or ``None`` when the value is not integer-like."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_settings_for_backend(settings: Any, backend_mode: str) -> List[str]:
    """Return human-readable preflight errors for starting ``backend_mode``.

    Parameters
    ----------
    settings:
        A :class:`QtSettingsState` (or any object exposing ``input_folder`` /
        ``output_folder`` / ``batch_size`` / ``drizzle_group_size``) or a
        :class:`~seestar.gui_qt.run_bridge.RunRequest` (whose ``backend_kwargs``
        carry the same values under ``input_dir`` / ``output_dir`` /
        ``batch_size`` / ``drizzle_group_size``).
    backend_mode:
        One of ``"simulated"`` / ``"seestar"``.  Only ``"seestar"`` is
        preflighted; any other mode returns ``[]`` (permissive).

    Returns
    -------
    list[str]
        Zero or more human-readable error strings.  An empty list means the
        settings are acceptable to start a real backend.
    """
    if backend_mode != SEESTAR_MODE:
        return []

    errors: List[str] = []

    input_folder = _read(settings, "input_folder", "input_dir", "")
    output_folder = _read(settings, "output_folder", "output_dir", "")
    if not _nonempty_str(input_folder):
        errors.append("Input folder is empty.")
    if not _nonempty_str(output_folder):
        errors.append("Output folder is empty.")

    batch_size = _read(settings, "batch_size", "batch_size", 0)
    batch_int = _as_int(batch_size)
    if batch_int is None:
        errors.append(f"Batch size must be an integer, got {batch_size!r}.")
    elif batch_int < -1:
        errors.append(
            f"Batch size must be -1 (auto) or greater, got {batch_int!r}."
        )

    drizzle_group_size = _read(
        settings, "drizzle_group_size", "drizzle_group_size", None
    )
    if drizzle_group_size is not None:
        group_int = _as_int(drizzle_group_size)
        if group_int is None:
            errors.append(
                f"Drizzle group size must be an integer, "
                f"got {drizzle_group_size!r}."
            )
        elif group_int <= 0:
            errors.append(
                f"Drizzle group size must be greater than 0, "
                f"got {group_int!r}."
            )

    return errors
