"""Qt-side run-settings handoff (M20 seam).

The canonical, shared builder :func:`seestar.gui.run_config.build_backend_kwargs`
deliberately does **not** emit ``use_gpu`` or ``max_hq_mem_gb``: the engine
reads those two settings from the ``SeestarQueuedStacker`` *instance*
(``use_gpu`` and ``max_hq_mem`` in bytes), never from a ``start_processing``
keyword.  That keeps the Tk run path byte-identical (it never carries these
fields in its snapshot).

The Qt shell collects ``use_gpu`` and ``max_hq_mem_gb`` in
:class:`~seestar.gui_qt.settings_state.QtSettingsState` (M8/M16/M19).  This
module turns those collected values into *seam-only* fields on a
:class:`~seestar.gui.run_config.RunRequest` — exactly the same pattern as the
existing ``stack_final_combine`` seam — so the Qt backend adapter
(:class:`~seestar.gui_qt.backend_runner.SeestarQueuedStackerBackend`) can apply
them to the stacker instance after ``split_backend_kwargs`` filters them out of
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
QT_SEAM_FIELDS = ("use_gpu", "max_hq_mem_gb", "reference_origin_hint")


def attach_run_settings(
    request: RunRequest,
    *,
    use_gpu: bool = False,
    max_hq_mem_gb: float = 8.0,
    reference_origin_hint: str | None = None,
) -> RunRequest:
    """Return a new ``RunRequest`` carrying the Qt-collected seam settings.

    The canonical ``request`` is never mutated: a fresh, still-immutable
    snapshot is built with ``use_gpu`` and ``max_hq_mem_gb`` appended to
    ``backend_kwargs``.  The defaults (``False`` / ``8.0``) match the Qt/Tk
    defaults, so a bare surface (no persisted settings, untouched controls)
    degrades to today's behaviour.
    """
    merged = dict(request.backend_kwargs)
    merged["use_gpu"] = bool(use_gpu)
    merged["max_hq_mem_gb"] = float(max_hq_mem_gb)
    merged["reference_origin_hint"] = reference_origin_hint
    return RunRequest(
        backend_kwargs=MappingProxyType(merged),
        align_on_disk=request.align_on_disk,
        special_single=request.special_single,
        resume_intent=request.resume_intent,
        resume_source=request.resume_source,
    )
