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

    # --- Colour / post-processing ---
    apply_chroma_correction: bool = True
    apply_final_scnr: bool = True
    final_scnr_target_channel: str = "green"
    final_scnr_amount: float = 0.6
    final_scnr_preserve_luminosity: bool = True
    bn_grid_size_str: str = "24x24"
    bn_perc_low: int = 5
    bn_perc_high: int = 40
    bn_std_factor: float = 1.5
    bn_min_gain: float = 0.2
    bn_max_gain: float = 7.0
    cb_border_size: int = 25
    cb_blur_radius: int = 8
    cb_min_b_factor: float = 0.4
    cb_max_b_factor: float = 1.5

    # --- Cropping ---
    apply_master_tile_crop: bool = False
    master_tile_crop_percent: float = 18.0
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

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow snapshot of the current state as a dict."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
