"""Qt-independent run-request snapshot.

This module is the first PySide6 migration seam (M0).  It turns validated GUI
state/settings into a plain, immutable, testable snapshot of everything the
scientific backend needs to start a run, so worker/starter threads no longer
have to re-read Tk (or later Qt) widgets to determine run configuration.

The module deliberately imports nothing GUI-related: no ``tkinter``, no Qt, no
``numpy``.  It only needs a plain "settings" object exposing the same attribute
names as :class:`seestar.gui.settings.SettingsManager` (duck typing), which lets
tests exercise the exact backend-kwargs mapping without a Tk root.

Semantics preserved (see P0 audit):

* ``batch_size`` < 0 / == 0 / == 1 / >= 2  (align-on-disk / chunk-size logic),
* drizzle ``Final`` / ``Incremental`` and ``drizzle_group_size`` propagation,
* solver preference ``none`` / ``astap`` / ``zesolver``,
* the exact backend ``start_processing(**kwargs)`` surface.

The values produced here are byte-for-byte equivalent to the inline
``backend_kwargs`` dictionary that previously lived inside
``SeestarStackerGUI.start_processing``'s ``_starter`` thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional


# Stable run-intent vocabulary (Resume Contract v2).  These two values are the
# *explicit* fresh/resume contract carried end-to-end: QtSettingsState ->
# RunRequest -> Qt backend adapter -> SeestarQueuedStacker.start_processing.
# The engine never derives intent from artifacts; this is the only source of
# truth for whether a run is a fresh stack or a resume.
RUN_INTENT_FRESH = "fresh"
RUN_INTENT_RESUME = "resume"


def normalize_run_intent(value: Any) -> str:
    """Coerce a run-intent value to ``"fresh"`` or ``"resume"``.

    Anything that is not exactly ``RUN_INTENT_RESUME`` (including ``None``,
    unknown/legacy spellings and non-strings) degrades to ``RUN_INTENT_FRESH``
    so a corrupt or absent intent can never silently turn a fresh start into a
    resume (fail-closed toward "do not touch existing state").
    """
    if value == RUN_INTENT_RESUME:
        return RUN_INTENT_RESUME
    return RUN_INTENT_FRESH


@dataclass(frozen=True)
class RunRequest:
    """Immutable snapshot handed from the GUI thread to the backend starter.

    Attributes
    ----------
    backend_kwargs:
        Read-only keyword mapping handed to the backend starter.  Every key
        except the seam-only fields (see ``SEAM_ONLY_KWARGS`` /
        ``split_backend_kwargs``) is forwarded verbatim to
        ``SeestarQueuedStacker.start_processing(**backend_kwargs)``; the
        seam-only ``stack_final_combine`` value is applied to the stacker
        instance instead.  Mutable settings-owned containers are shallow-copied
        while the snapshot is built, so later GUI/settings mutations cannot
        change the run request.
    align_on_disk:
        Value to assign to ``SeestarQueuedStacker.align_on_disk`` before the
        worker thread starts (``batch_size >= 1``).
    special_single:
        ``True`` when the batch_size==1 CSV single-batch mode was prepared by
        ``SeestarStackerGUI._prepare_single_batch_if_needed``.  Used by the GUI
        finish callback to re-sync UI variables.
    resume_intent:
        Explicit run intent: ``RUN_INTENT_FRESH`` or ``RUN_INTENT_RESUME``.
        Defaults to fresh; never derived from artifacts.
    resume_source:
        Optional explicit resume source path (``None`` for fresh; a resolvable
        output/run directory or Last Stack parent for resume).  Carried to the
        engine but not yet used for CFG discovery/restoration in this slice.
    """

    backend_kwargs: Mapping[str, Any]
    align_on_disk: bool
    special_single: bool = False
    resume_intent: str = RUN_INTENT_FRESH
    resume_source: Optional[str] = None


def compute_align_on_disk(batch_size: Any) -> bool:
    """Return the ``align_on_disk`` flag for a given (validated) batch size.

    Mirrors the previous inline logic exactly: ``int(batch_size) >= 1``, with a
    defensive fallback to ``False`` when the value cannot be coerced.
    """
    try:
        return int(batch_size) >= 1
    except Exception:
        return False


def _copy_list_or_none(value: Any) -> Any:
    """Return a shallow list copy for GUI-owned lists; preserve other values."""
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    return value


def _copy_dict_or_none(value: Any) -> Any:
    """Return a shallow dict copy for GUI-owned dicts; preserve other values."""
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    return value


# Snapshot fields carried in ``backend_kwargs`` for adapters/tests but *not*
# accepted by ``SeestarQueuedStacker.start_processing``.  The runner/adapter
# must filter these out before calling ``start_processing(**kwargs)`` and apply
# them to the stacker instance instead.
#
# * ``stack_final_combine`` — the QueueManager reads it from its instance (or
#   its settings object), never from a ``start_processing`` argument.
# * ``use_gpu`` / ``max_hq_mem_gb`` — Qt-collected seam fields (M20).  The
#   engine reads ``request_gpu`` (GPU acceleration intent, resolved through
#   ``AccelerationPolicy``) and ``max_hq_mem`` (bytes) from the stacker
#   instance, never from ``start_processing``; the Qt backend adapter applies
#   them to the stacker after this split.  They are *not* emitted by
#   ``build_backend_kwargs`` (so the Tk flow is byte-identical) — the Qt shell
#   attaches them to its ``RunRequest`` at the call site.
SEAM_ONLY_KWARGS = frozenset(
    {"stack_final_combine", "use_gpu", "max_hq_mem_gb", "reference_origin_hint"}
)


def split_backend_kwargs(
    backend_kwargs: Mapping[str, Any],
) -> tuple:
    """Partition a run snapshot into ``(start_kwargs, seam_kwargs)``.

    ``start_kwargs`` is safe to forward verbatim to
    ``SeestarQueuedStacker.start_processing(**start_kwargs)``; ``seam_kwargs``
    holds the snapshot fields (``stack_final_combine`` and the Qt-collected
    ``use_gpu`` / ``max_hq_mem_gb``) that must be applied to the stacker
    instance instead of being passed through as keyword arguments.
    """
    start_kwargs: Dict[str, Any] = {}
    seam_kwargs: Dict[str, Any] = {}
    for key, value in backend_kwargs.items():
        if key in SEAM_ONLY_KWARGS:
            seam_kwargs[key] = value
        else:
            start_kwargs[key] = value
    return start_kwargs, seam_kwargs


def build_backend_kwargs(
    settings: Any,
    initial_additional_folders: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Collect the full backend kwargs surface from a validated settings object.

    This is the authoritative mapping from GUI state/settings to the backend
    ``SeestarQueuedStacker.start_processing`` call.  It reads only plain
    attributes from ``settings`` (never widgets) and must stay in lock-step
    with the backend signature.
    """

    # ``match_background_for_final`` may be None (unset) or a truthy/falsy
    # value; preserve the exact previous handling.
    raw_match_bg = getattr(settings, "match_background_for_final", None)
    if raw_match_bg is None:
        match_background_for_final = None
    else:
        match_background_for_final = bool(raw_match_bg)

    winsor_raw = settings.stack_winsor_limits
    if isinstance(winsor_raw, str):
        winsor_limits = tuple(
            float(x.strip()) for x in winsor_raw.split(",")
        )
    else:
        winsor_limits = (0.05, 0.05)

    return {
        "input_dir": settings.input_folder,
        "output_dir": settings.output_folder,
        "temp_folder": settings.temp_folder,
        "output_filename": settings.output_filename,
        "reference_path_ui": settings.reference_image_path,
        "initial_additional_folders": _copy_list_or_none(initial_additional_folders),
        "stacking_mode": settings.stacking_mode,
        "kappa": settings.kappa,
        "stack_kappa_low": settings.stack_kappa_low,
        "stack_kappa_high": settings.stack_kappa_high,
        "winsor_limits": winsor_limits,
        "normalize_method": settings.stack_norm_method,
        "weighting_method": settings.stack_weight_method,
        "batch_size": settings.batch_size,
        "ordered_files": _copy_list_or_none(getattr(settings, "order_file_list", None)),
        "correct_hot_pixels": settings.correct_hot_pixels,
        "hot_pixel_threshold": settings.hot_pixel_threshold,
        "neighborhood_size": settings.neighborhood_size,
        "bayer_pattern": settings.bayer_pattern,
        "perform_cleanup": settings.cleanup_temp,
        "use_weighting": settings.stack_weight_method != "none",
        "weight_by_snr": settings.weight_by_snr,
        "weight_by_stars": settings.weight_by_stars,
        "snr_exp": settings.snr_exponent,
        "stars_exp": settings.stars_exponent,
        "min_w": settings.min_weight,
        "use_drizzle": settings.use_drizzle,
        "drizzle_scale": float(settings.drizzle_scale),
        "drizzle_wht_threshold": settings.drizzle_wht_threshold,
        "drizzle_mode": settings.drizzle_mode,
        "drizzle_kernel": settings.drizzle_kernel,
        "drizzle_pixfrac": settings.drizzle_pixfrac,
        "drizzle_group_size": settings.drizzle_group_size,
        "apply_chroma_correction": settings.apply_chroma_correction,
        "apply_final_scnr": settings.apply_final_scnr,
        "final_scnr_target_channel": settings.final_scnr_target_channel,
        "final_scnr_amount": settings.final_scnr_amount,
        "final_scnr_preserve_luminosity": settings.final_scnr_preserve_luminosity,
        "bn_grid_size_str": settings.bn_grid_size_str,
        "bn_perc_low": settings.bn_perc_low,
        "bn_perc_high": settings.bn_perc_high,
        "bn_std_factor": settings.bn_std_factor,
        "bn_min_gain": settings.bn_min_gain,
        "bn_max_gain": settings.bn_max_gain,
        "cb_border_size": settings.cb_border_size,
        "cb_blur_radius": settings.cb_blur_radius,
        "cb_min_b_factor": settings.cb_min_b_factor,
        "cb_max_b_factor": settings.cb_max_b_factor,
        "apply_master_tile_crop": settings.apply_master_tile_crop,
        "master_tile_crop_percent": settings.master_tile_crop_percent,
        "final_edge_crop_percent": settings.final_edge_crop_percent,
        "apply_photutils_bn": settings.apply_photutils_bn,
        "photutils_bn_box_size": settings.photutils_bn_box_size,
        "photutils_bn_filter_size": settings.photutils_bn_filter_size,
        "photutils_bn_sigma_clip": settings.photutils_bn_sigma_clip,
        "photutils_bn_exclude_percentile": settings.photutils_bn_exclude_percentile,
        "apply_feathering": settings.apply_feathering,
        "feather_blur_px": settings.feather_blur_px,
        "apply_batch_feathering": settings.apply_batch_feathering,
        "apply_coverage_render": getattr(settings, "apply_coverage_render", False),
        "apply_low_wht_mask": settings.apply_low_wht_mask,
        "low_wht_percentile": settings.low_wht_percentile,
        "low_wht_soften_px": settings.low_wht_soften_px,
        "is_mosaic_run": settings.mosaic_mode_active,
        "mosaic_settings": _copy_dict_or_none(settings.mosaic_settings),
        "astap_path": settings.astap_path,
        "astap_data_dir": settings.astap_data_dir,
        "local_solver_preference": settings.local_solver_preference,
        "astap_search_radius": settings.astap_search_radius,
        "astap_downsample": settings.astap_downsample,
        "astap_sensitivity": settings.astap_sensitivity,
        "save_as_float32": settings.save_final_as_float32,
        "preserve_linear_output": settings.preserve_linear_output,
        "stack_final_combine": settings.stack_final_combine,
        "reproject_between_batches": settings.reproject_between_batches,
        "reproject_coadd_final": settings.reproject_coadd_final,
        "match_background_for_final": match_background_for_final,
    }


def build_run_request(
    settings: Any,
    *,
    initial_additional_folders: Optional[List[str]] = None,
    auto_chunk_size: Optional[int] = None,
    special_single: bool = False,
    resume_intent: Optional[str] = None,
    resume_source: Optional[str] = None,
) -> RunRequest:
    """Build the full immutable run snapshot from validated settings.

    Parameters
    ----------
    settings:
        A validated settings object (typically a ``SettingsManager`` after
        ``update_from_ui`` + ``validate_settings``).  Only plain attributes are
        read.
    initial_additional_folders:
        Folders staged by the GUI to pass to the backend (may be ``None``).
    auto_chunk_size:
        Automatic chunk size (from system RAM) used for the non-CSV
        ``batch_size == 1`` path.  Mirrors ``_get_auto_chunk_size``.
    special_single:
        Whether the batch_size==1 CSV single-batch mode was already prepared.
        When ``False`` and ``batch_size == 1``, ``chunk_size`` is added to the
        backend kwargs exactly as before.
    resume_intent:
        Explicit run intent.  When ``None``, read from
        ``settings.resume_intent`` (falling back to fresh).  Normalised via
        :func:`normalize_run_intent`.
    resume_source:
        Explicit resume source path.  When ``None``, read from
        ``settings.resume_source`` (falling back to ``None``).
    """
    backend_kwargs = build_backend_kwargs(
        settings, initial_additional_folders=initial_additional_folders
    )

    batch_size = getattr(settings, "batch_size", 0)
    if batch_size == 1 and not special_single:
        backend_kwargs["chunk_size"] = auto_chunk_size

    if resume_intent is None:
        resume_intent = getattr(settings, "resume_intent", RUN_INTENT_FRESH)
    if resume_source is None:
        resume_source = getattr(settings, "resume_source", None) or None

    return RunRequest(
        backend_kwargs=MappingProxyType(backend_kwargs),
        align_on_disk=compute_align_on_disk(batch_size),
        special_single=special_single,
        resume_intent=normalize_run_intent(resume_intent),
        resume_source=resume_source,
    )
