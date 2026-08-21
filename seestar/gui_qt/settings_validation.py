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
  ``> 0``.  ``batch_size == 1`` is the historical boring/single-batch path and
  is **not** rejected here: the Qt real backend simply passes it through (its
  CSV single-batch handling is a later milestone, not a preflight concern).
  Callers normalise the UI ``0`` value via :func:`normalize_batch_size` before
  building the request.
* In ``backend_mode == "seestar"``, when either ``reproject_between_batches``
  or ``reproject_coadd_final`` is true, a *solver* is required before start
  (reproducing ``resolve_solver_gate`` semantics):

  - ``local_solver_preference == "none"`` (or any unknown value) is rejected;
  - ``"astap"`` requires a non-empty ``astap_path``;
  - ``"zesolver"`` is accepted when ZeSolver is operational **or** when an
    ASTAP fallback is configured (``astap_path`` non-empty).  The ZeSolver
    operational-readiness flag is injected by the caller (see
    :func:`seestar.gui_qt.solver_probe.probe_zesolver_operational`) because
    probing would require importing the engine, which is forbidden here.
  - Reprojection **off** never requires a solver.
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

# Solver preferences understood by the Qt real-backend reproject gate.
SOLVER_NONE = "none"
SOLVER_ASTAP = "astap"
SOLVER_ZESOLVER = "zesolver"


def normalize_batch_size(batch_size, reproject_coadd_final: bool = False):
    """Return the effective backend batch size for a UI batch-size value.

    Reproduces the historical Tk ``validate_settings`` contract so the Qt
    shell hands the backend the same sentinel/special values:

    * ``0`` + not ``reproject_coadd_final``  -> ``-1`` (Auto sentinel: the
      queue manager estimates the batch size dynamically),
    * ``0`` + ``reproject_coadd_final``      -> ``0``  (special batch-zero /
      "Reproject & Coadd" single in-memory batch — must NOT become Auto),
    * ``1``                                  -> ``1``  (boring/single-batch
      historical path — not refused by preflight),
    * ``>= 2``                               -> unchanged (explicit batch).

    Negative values (other than ``-1``) also become ``-1`` (Auto), matching the
    historical ``<= 0`` coercion.  Non-integer values pass through unchanged so
    validation can report them.
    """
    try:
        requested = int(batch_size)
    except (TypeError, ValueError):
        return batch_size
    allow_mode_zero = bool(reproject_coadd_final)
    if requested == 0 and allow_mode_zero:
        return 0
    if requested <= 0:
        return -1
    return requested


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


def _is_truthy(value: Any) -> bool:
    """Return a defensive boolean for a settings flag (``None``/``""`` -> False)."""
    return bool(value)


def _reproject_enabled(settings: Any) -> bool:
    """True when either reproject flag is truthy on the settings object."""
    between = _read(
        settings, "reproject_between_batches", "reproject_between_batches", False
    )
    coadd = _read(settings, "reproject_coadd_final", "reproject_coadd_final", False)
    return _is_truthy(between) or _is_truthy(coadd)


def _reproject_solver_errors(
    settings: Any, zesolver_operational: bool = False
) -> List[str]:
    """Return preflight errors for the reproject solver gate (seestar mode).

    Reproduces the historical ``resolve_solver_gate`` truth table (as defined
    in the core solver config), but is pure stdlib: it never imports the engine
    or the Tk GUI.  The ZeSolver operational-readiness probe stays outside this
    module (see :func:`seestar.gui_qt.solver_probe.probe_zesolver_operational`)
    and is injected here as a plain boolean so the validator stays unit-testable
    in complete isolation.

    * ``"zesolver"`` + ZeSolver operational             -> allowed (no ASTAP needed),
    * ``"zesolver"`` + ZeSolver unavailable + ASTAP     -> allowed (ASTAP fallback),
    * ``"zesolver"`` + ZeSolver unavailable + no ASTAP  -> blocked,
    * ``"astap"`` + ASTAP configured                    -> allowed,
    * ``"astap"`` + no ASTAP                            -> blocked,
    * ``"none"`` / unknown                              -> blocked (no solver).
    """
    errors: List[str] = []
    if not _reproject_enabled(settings):
        return errors

    raw_pref = _read(
        settings, "local_solver_preference", "local_solver_preference", SOLVER_NONE
    )
    pref = str(raw_pref or "").strip().lower()
    astap_path = str(
        _read(settings, "astap_path", "astap_path", "") or ""
    ).strip()
    astap_configured = bool(astap_path)

    if pref == SOLVER_ZESOLVER:
        if zesolver_operational or astap_configured:
            return errors
        errors.append(
            "Reprojection with ZeSolver requires ZeSolver to be operational "
            "or an ASTAP fallback (astap_path is empty); no usable solver "
            "is available."
        )
        return errors

    if pref == SOLVER_ASTAP:
        if astap_configured:
            return errors
        errors.append(
            "Reprojection requires ASTAP, but no ASTAP path is configured "
            "(astap_path is empty)."
        )
        return errors

    # "none" (default) or any unknown preference -> no usable solver selected.
    errors.append(
        "Reprojection requires a local astrometric solver, but "
        f"local_solver_preference is {pref!r} (expected 'astap' or 'zesolver')."
    )
    return errors


def validate_settings_for_backend(
    settings: Any,
    backend_mode: str,
    *,
    zesolver_operational: bool = False,
) -> List[str]:
    """Return human-readable preflight errors for starting ``backend_mode``.

    Parameters
    ----------
    settings:
        A :class:`QtSettingsState` (or any object exposing ``input_folder`` /
        ``output_folder`` / ``batch_size`` / ``drizzle_group_size`` /
        ``reproject_between_batches`` / ``reproject_coadd_final`` /
        ``local_solver_preference`` / ``astap_path``) or a
        :class:`~seestar.gui_qt.run_bridge.RunRequest` (whose ``backend_kwargs``
        carry the same values under ``input_dir`` / ``output_dir`` /
        ``batch_size`` / ``drizzle_group_size`` / ``reproject_between_batches``
        / ``reproject_coadd_final`` / ``local_solver_preference`` /
        ``astap_path``).
    backend_mode:
        One of ``"simulated"`` / ``"seestar"``.  Only ``"seestar"`` is
        preflighted; any other mode returns ``[]`` (permissive).
    zesolver_operational:
        Whether the ZeSolver public API reports itself operational.  Only used
        by the reproject solver gate when ``local_solver_preference ==
        "zesolver"``.  Defaults to ``False`` (the probe is left to the caller).

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

    # Reproject solver gate: only enforced for the real backend, and only when
    # reprojection is actually requested (a non-reprojecting run needs no
    # solver).
    errors.extend(_reproject_solver_errors(settings, zesolver_operational))

    return errors
