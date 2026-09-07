"""Qt-side run-settings handoff (M20 seam).

The canonical, shared builder :func:`seestar.gui.run_config.build_backend_kwargs`
deliberately does **not** emit ``use_gpu`` or ``max_hq_mem_gb``: the engine
reads those two settings from the ``SeestarQueuedStacker`` *instance*
(``use_gpu`` and ``max_hq_mem`` in bytes), never from a ``start_processing``
keyword.  That keeps the Tk run path byte-identical (it never carries these
fields in its snapshot).

8.4.0 stage E2: ``max_hq_mem_gb`` is NO LONGER a seam field.  AUTO is the
product CPU memory policy (the stage E1 engine resolves the budget at
execution); a persisted legacy HQ RAM value is read for migration/diagnostics
only and must never become the runtime budget.  An expert budget only enters
through the explicit ``ZSSS_CPU_MEMORY_OVERRIDE_BYTES`` env seam (never as a
normal GUI knob or run-request field).

The Qt shell collects ``use_gpu`` in
:class:`~seestar.gui_qt.settings_state.QtSettingsState` (M8/M16/M19).  This
module turns the collected value into a *seam-only* field on a
:class:`~seestar.gui.run_config.RunRequest` — exactly the same pattern as the
existing ``stack_final_combine`` seam — so the Qt backend adapter
(:class:`~seestar.gui_qt.backend_runner.SeestarQueuedStackerBackend`) can apply
it to the stacker instance after ``split_backend_kwargs`` filters it out of
the ``start_processing`` surface.

Import-hygiene: this module imports nothing GUI-, Tk- or engine-related — only
:class:`RunRequest` from the canonical builder plus ``types.MappingProxyType``.
"""

from __future__ import annotations

from types import MappingProxyType

from .run_bridge import RunRequest

# Seam-only fields the Qt shell attaches to its RunRequest.  These mirror the
# entries in ``seestar.gui.run_config.SEAM_ONLY_KWARGS`` (they are filtered out
# of ``start_processing`` kwargs by ``split_backend_kwargs``).
QT_SEAM_FIELDS = ("use_gpu", "reference_origin_hint")


def attach_run_settings(
    request: RunRequest,
    *,
    use_gpu: bool = False,
    reference_origin_hint: str | None = None,
) -> RunRequest:
    """Return a new ``RunRequest`` carrying the Qt-collected seam settings.

    The canonical ``request`` is never mutated: a fresh, still-immutable
    snapshot is built with ``use_gpu`` appended to ``backend_kwargs``.  The
    default (``False``) matches the Qt/Tk default, so a bare surface (no
    persisted settings, untouched controls) degrades to today's behaviour.
    8.4.0 stage E2: ``max_hq_mem_gb`` is intentionally absent — the legacy
    value never reaches the run budget.
    """
    merged = dict(request.backend_kwargs)
    merged["use_gpu"] = bool(use_gpu)
    merged["reference_origin_hint"] = reference_origin_hint
    return RunRequest(
        backend_kwargs=MappingProxyType(merged),
        align_on_disk=request.align_on_disk,
        special_single=request.special_single,
        resume_intent=request.resume_intent,
        resume_source=request.resume_source,
    )
