"""Qt-side settings state (M2 seam).

This module defines :class:`QtSettingsState`, the plain, toolkit-agnostic model
that the PySide6 shell uses to hold validated run settings.  It deliberately
mirrors the attribute surface read by :mod:`seestar.gui.run_config` (the
Qt/Tk-independent run-request builder), so a ``QtSettingsState`` instance can
be handed straight to ``build_run_request``/``build_backend_kwargs`` without
any Tk or Qt object ever crossing the seam.

Design constraints honoured here:

* **No Qt, no Tk, no numpy, no engine imports.**  This module is pure stdlib so
  it can be imported and unit-tested in complete isolation.
* **Defaults aligned with the Tk settings manager** (``SettingsManager``
  ``get_default_values``) for every attribute that also exists there.  The
  alignment is verified by tests that load the two modules side by side.
* **Full attribute surface.**  Every attribute read by
  ``build_backend_kwargs`` (plus ``batch_size``) is present with a safe default,
  so building a ``RunRequest`` from a default instance never raises
  ``AttributeError``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List


# Closed vocabulary for the UI language field (M9).  An unknown persisted value
# degrades to English so a corrupt settings file can never leave the shell in an
# unrecognised language.  Kept inline (not imported from ``gui_qt.localization``)
# so this module stays a pure, dependency-free stdlib model.
_SUPPORTED_LANGUAGES = ("en", "fr")


def _default_mosaic_settings() -> Dict[str, Any]:
    """Return a fresh copy of the default mosaic settings dict."""
    return {
        "kernel": "square",
        "pixfrac": 0.8,
        "use_gpu": False,
        "fillval": "0.0",
        "wht_threshold": 0.01,
        "alignment_mode": "local_fast_fallback",
        "fastalign_orb_features": 3000,
        "fastalign_min_abs_matches": 8,
        "fastalign_min_ransac": 4,
        "fastalign_ransac_thresh": 2.5,
        "fastalign_dao_fwhm": 3.5,
        "fastalign_dao_thr_sig": 8.0,
        "fastalign_dao_max_stars": 750,
        "mosaic_scale_factor": 1,
    }


def _coerce_value(raw: Any, default: Any) -> Any:
    """Coerce ``raw`` toward the type/shape of ``default``; fall back to it.

    This is the defensive, pure-stdlib coercion used by
    :meth:`QtSettingsState.from_dict` so a partially-corrupt persisted value
    for a known field degrades to the code default instead of raising.  The
    rules are deliberately conservative:

    * ``None`` default (``match_background_for_final``): accept ``None``, a
      real ``bool``, or the canonical ``"true"``/``"false"`` string spellings.
    * ``bool`` default: accept a real ``bool`` or the ints ``0``/``1``.
    * ``int``/``float`` defaults: accept a number (or numeric string) that is
      not a ``bool``; ``int`` goes through ``float`` so ``"3.0"`` works.
    * ``str`` default: accept a ``str`` only (``None`` -> default).
    * ``list``/``dict`` defaults: accept the matching container; ``dict`` is
      merged onto a default copy so missing sub-keys are filled.
    """
    if default is None:
        if raw is None:
            return None
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            spelling = raw.strip().lower()
            if spelling in ("true", "1"):
                return True
            if spelling in ("false", "0"):
                return False
        return None
    if isinstance(default, bool):
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return bool(raw)
        return default
    if isinstance(default, int):
        if isinstance(raw, bool):
            return default
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        if isinstance(raw, bool):
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default
    if isinstance(default, str):
        if isinstance(raw, str):
            return raw
        return default
    if isinstance(default, list):
        if isinstance(raw, list):
            return list(raw)
        return list(default)
    if isinstance(default, dict):
        if isinstance(raw, dict):
            merged = dict(default)
            for key, value in raw.items():
                if key in default:
                    merged[key] = _coerce_value(value, default[key])
                else:
                    # Preserve unknown nested keys for forward compatibility,
                    # but never let a bad value for a known nested key crash
                    # widget application later.
                    merged[key] = value
            return merged
        return dict(default)
    return default


@dataclass
class QtSettingsState:
    """Plain, mutable settings model for the PySide6 shell.

    The fields below are the *complete* surface consumed by
    ``seestar.gui.run_config.build_backend_kwargs``.  Values are plain Python
    types (``str``/``int``/``float``/``bool``/``list``/``dict``/``None``); the
    GUI shell is responsible for reading them out of Qt widgets, and
    ``run_config`` is responsible for turning them into backend kwargs.
    """

    # --- Input / output / temp paths ---
    input_folder: str = ""
    output_folder: str = ""
    temp_folder: str = ""
    output_filename: str = ""
    reference_image_path: str = ""
    # Last completed stack path (GUI parity only; the backend does not read it
    # yet — full resume semantics are a later milestone).
    last_stack_path: str = ""

    # --- Stacking method ---
    stacking_mode: str = "kappa-sigma"
    kappa: float = 2.5
    stack_kappa_low: float = 3.0
    stack_kappa_high: float = 3.0
    stack_winsor_limits: str = "0.05,0.05"
    stack_norm_method: str = "none"
    stack_weight_method: str = "none"
    # Final-combination business choice.  One of
    # ``mean`` / ``median`` / ``winsorized_sigma_clip`` / ``reproject`` /
    # ``reproject_coadd``.  Drives the two reproject flags below exactly like
    # the Tk ``SettingsManager.update_from_ui`` derivation.
    stack_final_combine: str = "mean"
    batch_size: int = 0
    # HQ RAM limit (GB) for the single-batch / boring stack subprocess (Tk
    # ``max_hq_mem_var``).  GUI parity only today: the Qt backend bridge does
    # not consume it yet (the boring CLI ``--max-mem`` stays fixed at the 8.0
    # default) — backend E2E later.
    max_hq_mem_gb: float = 8.0
    order_file_list: List[str] = field(default_factory=list)

    # --- Hot pixel / calibration ---
    correct_hot_pixels: bool = True
    hot_pixel_threshold: float = 3.0
    neighborhood_size: int = 5
    bayer_pattern: str = "GRBG"
    cleanup_temp: bool = True

    # --- Quality weighting ---
    weight_by_snr: bool = True
    weight_by_stars: bool = True
    snr_exponent: float = 1.8
    stars_exponent: float = 0.5
    min_weight: float = 0.01

    # --- Drizzle ---
    use_drizzle: bool = False
    drizzle_scale: int = 2
    drizzle_wht_threshold: float = 0.7
    drizzle_mode: str = "Final"
    drizzle_kernel: str = "square"
    drizzle_pixfrac: float = 1.0
    drizzle_group_size: int = 50
    # Drizzle GPU toggle (Tk ``use_gpu_var``, Stacking tab).  GUI parity only
    # today: ``build_backend_kwargs`` does not consume it (backend E2E later).
    use_gpu: bool = False

    # --- Colour / post-processing ---
    apply_chroma_correction: bool = True
    apply_final_scnr: bool = True
    final_scnr_target_channel: str = "green"
    final_scnr_amount: float = 0.6
    final_scnr_preserve_luminosity: bool = True
    # Expert-tab enable flags (BN / CB / final crop).  These are GUI gating
    # controls only: they enable/disable their sub-option widgets and are
    # persisted/restored like the Tk ``apply_bn_var`` / ``apply_cb_var`` /
    # ``apply_final_crop_var``, but they are NOT consumed by
    # ``build_backend_kwargs`` (the engine does not read them today).
    apply_bn: bool = True
    bn_grid_size_str: str = "24x24"
    bn_perc_low: int = 5
    bn_perc_high: int = 40
    bn_std_factor: float = 1.5
    bn_min_gain: float = 0.2
    bn_max_gain: float = 7.0
    apply_cb: bool = True
    cb_border_size: int = 25
    cb_blur_radius: int = 8
    cb_min_b_factor: float = 0.4
    cb_max_b_factor: float = 1.5

    # --- Cropping ---
    apply_master_tile_crop: bool = False
    master_tile_crop_percent: float = 18.0
    apply_final_crop: bool = True
    final_edge_crop_percent: float = 2.0

    # --- Photutils background normalisation ---
    apply_photutils_bn: bool = False
    photutils_bn_box_size: int = 128
    photutils_bn_filter_size: int = 11
    photutils_bn_sigma_clip: float = 3.0
    photutils_bn_exclude_percentile: float = 95.0

    # --- Feathering / low-weight mask ---
    apply_feathering: bool = True
    feather_blur_px: int = 256
    apply_batch_feathering: bool = True
    apply_low_wht_mask: bool = False
    low_wht_percentile: int = 5
    low_wht_soften_px: int = 128

    # --- Mosaic ---
    mosaic_mode_active: bool = False
    mosaic_settings: Dict[str, Any] = field(default_factory=_default_mosaic_settings)

    # --- Local solver ---
    astap_path: str = ""
    astap_data_dir: str = ""
    local_solver_preference: str = "none"
    astap_search_radius: float = 3.0
    astap_downsample: int = 1
    astap_sensitivity: int = 100

    # --- Output format / reprojection ---
    save_final_as_float32: bool = False
    preserve_linear_output: bool = False
    reproject_between_batches: bool = False
    reproject_coadd_final: bool = False

    # --- Final background matching (may be None when unset) ---
    match_background_for_final: Any = None

    # --- UI language (M9; persisted via the M8 settings JSON round-trip) ---
    language: str = "en"

    @classmethod
    def defaults(cls) -> Dict[str, Any]:
        """Return a fresh dict of field name -> default value.

        Mutable defaults are rebuilt via their ``default_factory`` so callers
        always get independent containers.
        """
        result: Dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                result[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
                result[f.name] = f.default_factory()  # type: ignore[misc]
            else:
                result[f.name] = None
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QtSettingsState":
        """Build a state from a persisted dict, ignoring unknown keys.

        Every known field present in ``data`` is coerced toward its default
        type via :func:`_coerce_value`; fields absent from ``data`` (or with an
        uncoercible value) fall back to the dataclass default.  Unknown keys
        are ignored so a future/legacy settings file never raises.  A non-dict
        ``data`` yields a pure default instance.
        """
        state = cls()
        if not isinstance(data, dict):
            return state
        defaults = cls.defaults()
        for f in dataclasses.fields(cls):
            name = f.name
            if name in data:
                setattr(state, name, _coerce_value(data[name], defaults[name]))
        # Normalise the UI language field: a supported code is kept, anything
        # else (unknown, corrupt, or non-string) falls back to English.
        if state.language not in _SUPPORTED_LANGUAGES:
            state.language = "en"
        return state

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow snapshot of the current state as a dict."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
