"""M0 seam tests: ``seestar.gui.run_config`` backend-kwargs parity.

``run_config`` is the Qt-independent boundary between GUI state/settings and
the scientific backend.  These tests verify, without a Tk root, that the
snapshot builder reproduces the exact ``backend_kwargs`` surface and the
batch-size / drizzle / solver semantics required by the P0 audit.

Both modules are loaded standalone by file path so the heavy ``seestar``
package tree (which needs optional deps) is not pulled in.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Load seestar.gui.run_config (pure stdlib, no Tk) ---
run_config_spec = importlib.util.spec_from_file_location(
    "seestar_run_config", ROOT / "seestar" / "gui" / "run_config.py"
)
run_config = importlib.util.module_from_spec(run_config_spec)
sys.modules["seestar_run_config"] = run_config
run_config_spec.loader.exec_module(run_config)

build_backend_kwargs = run_config.build_backend_kwargs
build_run_request = run_config.build_run_request
compute_align_on_disk = run_config.compute_align_on_disk
split_backend_kwargs = run_config.split_backend_kwargs
RunRequest = run_config.RunRequest

# --- Load seestar.gui.settings standalone (for a real SettingsManager) ---
# Loaded under a flat unique module name so the heavy/real ``seestar`` package
# tree (and other tests' ``seestar.gui`` stubs) are never touched.
settings_spec = importlib.util.spec_from_file_location(
    "seestar_settings_manager", ROOT / "seestar" / "gui" / "settings.py"
)
settings_mod = importlib.util.module_from_spec(settings_spec)
sys.modules["seestar_settings_manager"] = settings_mod
settings_spec.loader.exec_module(settings_mod)

SettingsManager = settings_mod.SettingsManager

# Backend kwarg -> settings attribute (verbatim pass-through).  Keys not listed
# here are derived/transformed (use_weighting, winsor_limits, drizzle_scale,
# match_background_for_final) or are caller-provided (initial_additional_folders).
DIRECT_MAP = {
    "input_dir": "input_folder",
    "output_dir": "output_folder",
    "temp_folder": "temp_folder",
    "output_filename": "output_filename",
    "reference_path_ui": "reference_image_path",
    "stacking_mode": "stacking_mode",
    "kappa": "kappa",
    "stack_kappa_low": "stack_kappa_low",
    "stack_kappa_high": "stack_kappa_high",
    "normalize_method": "stack_norm_method",
    "weighting_method": "stack_weight_method",
    "batch_size": "batch_size",
    "ordered_files": "order_file_list",
    "correct_hot_pixels": "correct_hot_pixels",
    "hot_pixel_threshold": "hot_pixel_threshold",
    "neighborhood_size": "neighborhood_size",
    "bayer_pattern": "bayer_pattern",
    "perform_cleanup": "cleanup_temp",
    "weight_by_snr": "weight_by_snr",
    "weight_by_stars": "weight_by_stars",
    "snr_exp": "snr_exponent",
    "stars_exp": "stars_exponent",
    "min_w": "min_weight",
    "use_drizzle": "use_drizzle",
    "drizzle_wht_threshold": "drizzle_wht_threshold",
    "drizzle_mode": "drizzle_mode",
    "drizzle_kernel": "drizzle_kernel",
    "drizzle_pixfrac": "drizzle_pixfrac",
    "drizzle_group_size": "drizzle_group_size",
    "apply_chroma_correction": "apply_chroma_correction",
    "apply_final_scnr": "apply_final_scnr",
    "final_scnr_target_channel": "final_scnr_target_channel",
    "final_scnr_amount": "final_scnr_amount",
    "final_scnr_preserve_luminosity": "final_scnr_preserve_luminosity",
    "bn_grid_size_str": "bn_grid_size_str",
    "bn_perc_low": "bn_perc_low",
    "bn_perc_high": "bn_perc_high",
    "bn_std_factor": "bn_std_factor",
    "bn_min_gain": "bn_min_gain",
    "bn_max_gain": "bn_max_gain",
    "cb_border_size": "cb_border_size",
    "cb_blur_radius": "cb_blur_radius",
    "cb_min_b_factor": "cb_min_b_factor",
    "cb_max_b_factor": "cb_max_b_factor",
    "apply_master_tile_crop": "apply_master_tile_crop",
    "master_tile_crop_percent": "master_tile_crop_percent",
    "final_edge_crop_percent": "final_edge_crop_percent",
    "apply_photutils_bn": "apply_photutils_bn",
    "photutils_bn_box_size": "photutils_bn_box_size",
    "photutils_bn_filter_size": "photutils_bn_filter_size",
    "photutils_bn_sigma_clip": "photutils_bn_sigma_clip",
    "photutils_bn_exclude_percentile": "photutils_bn_exclude_percentile",
    "apply_feathering": "apply_feathering",
    "feather_blur_px": "feather_blur_px",
    "apply_batch_feathering": "apply_batch_feathering",
    "apply_low_wht_mask": "apply_low_wht_mask",
    "low_wht_percentile": "low_wht_percentile",
    "low_wht_soften_px": "low_wht_soften_px",
    "is_mosaic_run": "mosaic_mode_active",
    "mosaic_settings": "mosaic_settings",
    "astap_path": "astap_path",
    "astap_data_dir": "astap_data_dir",
    "local_solver_preference": "local_solver_preference",
    "astap_search_radius": "astap_search_radius",
    "astap_downsample": "astap_downsample",
    "astap_sensitivity": "astap_sensitivity",
    "save_as_float32": "save_final_as_float32",
    "preserve_linear_output": "preserve_linear_output",
    "reproject_between_batches": "reproject_between_batches",
    "reproject_coadd_final": "reproject_coadd_final",
    "stack_final_combine": "stack_final_combine",
}

EXPECTED_BASE_KEYS = frozenset(
    list(DIRECT_MAP.keys())
    + [
        "initial_additional_folders",
        "use_weighting",
        "winsor_limits",
        "drizzle_scale",
        "match_background_for_final",
    ]
)


def _make_sm():
    return SettingsManager(settings_file="unused.json")


def test_backend_kwargs_key_surface():
    sm = _make_sm()
    kwargs = build_backend_kwargs(sm)
    assert set(kwargs.keys()) == EXPECTED_BASE_KEYS
    # chunk_size is only added by build_run_request for batch_size == 1.
    assert "chunk_size" not in kwargs


def test_direct_attribute_passthrough():
    """Every direct-mapped kwarg must be copied verbatim from settings."""
    sm = _make_sm()
    sentinels = {}
    for key, attr in DIRECT_MAP.items():
        sentinel = object()
        sentinels[key] = sentinel
        setattr(sm, attr, sentinel)

    kwargs = build_backend_kwargs(sm)
    for key, attr in DIRECT_MAP.items():
        assert kwargs[key] is sentinels[key], f"{key} != {attr}"


def test_stack_final_combine_preserved_in_backend_kwargs():
    """The selected final-combine key is carried in the run snapshot."""
    sm = _make_sm()
    for key in (
        "mean",
        "median",
        "winsorized_sigma_clip",
        "reproject",
        "reproject_coadd",
    ):
        sm.stack_final_combine = key
        assert build_backend_kwargs(sm)["stack_final_combine"] == key


def test_split_backend_kwargs_partitions_seam_only_field():
    """``split_backend_kwargs`` filters seam-only fields out of start kwargs."""
    sm = _make_sm()
    sm.stack_final_combine = "median"
    kwargs = build_backend_kwargs(sm)
    start_kwargs, seam_kwargs = split_backend_kwargs(kwargs)

    assert "stack_final_combine" not in start_kwargs
    assert seam_kwargs == {"stack_final_combine": "median"}
    # Everything else is forwarded verbatim (same object identity).
    for key, value in kwargs.items():
        if key != "stack_final_combine":
            assert start_kwargs[key] is value


def test_use_weighting_derived():
    sm = _make_sm()
    sm.stack_weight_method = "none"
    assert build_backend_kwargs(sm)["use_weighting"] is False
    sm.stack_weight_method = "snr"
    assert build_backend_kwargs(sm)["use_weighting"] is True


def test_winsor_limits_string_parsed():
    sm = _make_sm()
    sm.stack_winsor_limits = "0.10,0.20"
    assert build_backend_kwargs(sm)["winsor_limits"] == (0.10, 0.20)


def test_winsor_limits_non_string_fallback():
    sm = _make_sm()
    sm.stack_winsor_limits = (0.05, 0.05)  # already a tuple -> fallback
    assert build_backend_kwargs(sm)["winsor_limits"] == (0.05, 0.05)


def test_drizzle_scale_coerced_to_float():
    sm = _make_sm()
    sm.drizzle_scale = "2"
    assert build_backend_kwargs(sm)["drizzle_scale"] == 2.0
    assert isinstance(build_backend_kwargs(sm)["drizzle_scale"], float)


def test_match_background_none_and_bool():
    sm = _make_sm()
    sm.match_background_for_final = None
    assert build_backend_kwargs(sm)["match_background_for_final"] is None
    sm.match_background_for_final = 1
    assert build_backend_kwargs(sm)["match_background_for_final"] is True
    sm.match_background_for_final = 0
    assert build_backend_kwargs(sm)["match_background_for_final"] is False


def test_initial_additional_folders_passthrough():
    sm = _make_sm()
    folders = ["/a", "/b"]
    kwargs = build_backend_kwargs(sm, initial_additional_folders=folders)
    assert kwargs["initial_additional_folders"] == folders
    assert kwargs["initial_additional_folders"] is not folders
    assert build_backend_kwargs(sm)["initial_additional_folders"] is None


def test_compute_align_on_disk_semantics():
    # Mirrors the original inline ``int(batch_size) >= 1`` exactly: negative
    # (auto) sizes do NOT enable on-disk alignment at the GUI starter.
    assert compute_align_on_disk(-1) is False
    assert compute_align_on_disk(0) is False
    assert compute_align_on_disk(1) is True
    assert compute_align_on_disk(2) is True
    assert compute_align_on_disk("garbage") is False


def test_run_request_batch_size_edge_cases():
    # batch_size == 0 (in-memory single batch): no chunk_size, align_on_disk False
    sm = _make_sm()
    sm.batch_size = 0
    req = build_run_request(sm, auto_chunk_size=42)
    assert req.align_on_disk is False
    assert "chunk_size" not in req.backend_kwargs

    # batch_size >= 2 (normal batched): align_on_disk True, no chunk_size
    sm.batch_size = 5
    req = build_run_request(sm, auto_chunk_size=42)
    assert req.align_on_disk is True
    assert "chunk_size" not in req.backend_kwargs

    # batch_size == 1 without CSV prep: chunk_size added, align_on_disk True
    sm.batch_size = 1
    req = build_run_request(sm, auto_chunk_size=42, special_single=False)
    assert req.align_on_disk is True
    assert req.backend_kwargs["chunk_size"] == 42

    # batch_size == 1 with CSV prep (special_single): no chunk_size
    req = build_run_request(sm, auto_chunk_size=42, special_single=True)
    assert req.align_on_disk is True
    assert "chunk_size" not in req.backend_kwargs


def test_drizzle_and_solver_propagation():
    sm = _make_sm()
    sm.use_drizzle = True
    sm.drizzle_mode = "Incremental"
    sm.drizzle_group_size = 77
    sm.local_solver_preference = "zesolver"
    sm.astap_search_radius = 12.5

    kwargs = build_backend_kwargs(sm)
    assert kwargs["use_drizzle"] is True
    assert kwargs["drizzle_mode"] == "Incremental"
    assert kwargs["drizzle_group_size"] == 77
    assert kwargs["local_solver_preference"] == "zesolver"
    assert kwargs["astap_search_radius"] == 12.5


def test_run_request_is_immutable_dataclass():
    sm = _make_sm()
    req = build_run_request(sm)
    assert isinstance(req, RunRequest)
    try:
        req.align_on_disk = True
        raised = False
    except Exception:
        raised = True
    assert raised
    try:
        req.backend_kwargs["batch_size"] = 123
        raised = False
    except TypeError:
        raised = True
    assert raised


def test_run_request_copies_gui_owned_mutable_containers():
    sm = _make_sm()
    sm.order_file_list = ["a.fit", "b.fit"]
    sm.mosaic_settings = {"alignment_mode": "local"}
    folders = ["/extra-a", "/extra-b"]

    req = build_run_request(sm, initial_additional_folders=folders)

    assert req.backend_kwargs["ordered_files"] == ["a.fit", "b.fit"]
    assert req.backend_kwargs["ordered_files"] is not sm.order_file_list
    assert req.backend_kwargs["mosaic_settings"] == {"alignment_mode": "local"}
    assert req.backend_kwargs["mosaic_settings"] is not sm.mosaic_settings
    assert req.backend_kwargs["initial_additional_folders"] == folders
    assert req.backend_kwargs["initial_additional_folders"] is not folders

    sm.order_file_list.append("late.fit")
    sm.mosaic_settings["alignment_mode"] = "changed"
    folders.append("/late")

    assert req.backend_kwargs["ordered_files"] == ["a.fit", "b.fit"]
    assert req.backend_kwargs["mosaic_settings"] == {"alignment_mode": "local"}
    assert req.backend_kwargs["initial_additional_folders"] == ["/extra-a", "/extra-b"]


def test_run_request_against_real_settings_defaults():
    """Snapshot builds cleanly from a real SettingsManager (defaults)."""
    sm = _make_sm()
    sm.validate_settings()
    req = build_run_request(sm, initial_additional_folders=[])
    assert dict(req.backend_kwargs)
    assert req.backend_kwargs["batch_size"] == -1  # default 0 -> auto sentinel


def test_main_window_call_site_wires_run_request_arguments():
    """Guard the GUI call-site seam without importing Tk or building widgets."""
    src = (ROOT / "seestar" / "gui" / "main_window.py").read_text(encoding="utf-8")
    start = src.index("run_request = build_run_request(")
    call = src[start : start + 300]
    assert "self.settings" in call
    assert "initial_additional_folders=folders_to_pass_to_backend" in call
    assert "auto_chunk_size=self._get_auto_chunk_size()" in call
    assert "special_single=special_single" in call
