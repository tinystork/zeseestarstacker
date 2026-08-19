"""Native ZeSeestarStacker solver/stacking configuration module.

This module replaces the former embedded ``zemosaic.zemosaic_config`` copy
and exposes the same configuration contract (``DEFAULT_CONFIG`` plus the
``get_*`` / ``load_config`` / ``save_config`` helpers) while storing the
persisted configuration in a platform-aware *user* directory instead of next
to the source checkout.

Soft migration: if a legacy ``zemosaic_config.json`` file still exists at the
old project location, it is read (read-only) to seed defaults.  ``save_config``
always writes to the new user location and never touches the legacy file.
"""

import json
import os
import platform
from pathlib import Path

# Legacy filename used by the removed ZeMosaic copy.  Kept only for the
# soft-migration path (read-only) so existing user settings are not lost.
LEGACY_CONFIG_FILE_NAME = "zemosaic_config.json"

# New native configuration filename, stored in a platform-aware user directory.
CONFIG_FILE_NAME = "seestar_config.json"

DEFAULT_CONFIG = {
    "astap_executable_path": "",
    "astap_data_directory_path": "",
    "astap_default_search_radius": 3.0,
    "astap_default_downsample": 2,
    "astap_default_sensitivity": 100,
    "language": "en",
    "num_processing_workers": -1,  # -1 pour auto
    "stacking_normalize_method": "linear_fit",
    "stacking_weighting_method": "noise_variance",
    "stacking_rejection_algorithm": "winsorized_sigma_clip",
    "stacking_kappa_low": 3.0,
    "stacking_kappa_high": 3.0,
    "stacking_winsor_limits": "0.05,0.05",  # String, sera parsé
    "stacking_final_combine_method": "mean",
    "apply_radial_weight": False,
    "radial_feather_fraction": 0.8,
    "radial_shape_power": 2.0,
    "use_gpu_phase5": False,
    "gpu_id_phase5": 0,
    "gpu_selector": "",
    "final_assembly_method": "reproject_coadd",  # "reproject_coadd" | "incremental"
    "solver_method": "ansvr",
    "astrometry_local_path": "",
    "astrometry_api_key": "",
    "save_final_as_uint16": False,
    "coadd_use_memmap": True,
    "coadd_memmap_dir": "",
    "coadd_cleanup_memmap": True,
    "assembly_process_workers": 0,  # Worker count for final assembly (both methods)
    "auto_limit_frames_per_master_tile": True,
    "winsor_worker_limit": 4,
    "max_raw_per_master_tile": 0,
    # --- CLES POUR LE ROGNAGE DES MASTER TUILES ---
    "apply_master_tile_crop": True,
    "master_tile_crop_percent": 18.0,
    # --- FIN CLES POUR LE ROGNAGE ---
}


def _user_config_dir() -> str:
    """Return the platform-aware user configuration directory.

    Windows: ``%APPDATA%\\ZeSeestarStacker``
    macOS:   ``~/Library/Application Support/ZeSeestarStacker``
    Linux:   ``$XDG_CONFIG_HOME/ZeSeestarStacker`` (default ``~/.config``).
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Roaming"
        )
        return os.path.join(base, "ZeSeestarStacker")
    if platform.system() == "Darwin":
        base = os.path.join(
            os.path.expanduser("~"), "Library", "Application Support"
        )
        return os.path.join(base, "ZeSeestarStacker")
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return os.path.join(base, "ZeSeestarStacker")


def get_config_path() -> str:
    """Return the path of the user configuration file."""
    return os.path.join(_user_config_dir(), CONFIG_FILE_NAME)


def _legacy_config_candidates() -> list:
    """Return candidate paths for the legacy ``zemosaic_config.json`` file.

    The legacy configuration lived at the project root (or the current working
    directory when launched from a checkout).  These locations are consulted
    read-only for soft migration.
    """
    candidates: list = []
    try:
        project_root = Path(__file__).resolve().parents[2]
        candidates.append(project_root / LEGACY_CONFIG_FILE_NAME)
    except Exception:
        pass
    try:
        cwd_legacy = Path.cwd() / LEGACY_CONFIG_FILE_NAME
        if cwd_legacy not in candidates:
            candidates.append(cwd_legacy)
    except Exception:
        pass
    return candidates


def _read_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_loaded(current_config: dict, loaded_config: dict) -> dict:
    """Merge a loaded JSON object over ``current_config`` for known keys only."""
    for key in DEFAULT_CONFIG:
        if key in loaded_config:
            current_config[key] = loaded_config[key]
    return current_config


def load_config() -> dict:
    """Load the configuration, seeded from defaults, legacy file, then user file."""
    current_config = DEFAULT_CONFIG.copy()
    config_path = get_config_path()

    if os.path.exists(config_path):
        try:
            current_config = _merge_loaded(current_config, _read_json(config_path))
        except (json.JSONDecodeError, OSError):
            # Malformed user config -> keep defaults (do not fall back to legacy).
            pass
    else:
        # Soft migration: read legacy zemosaic_config.json (read-only) to seed
        # defaults.  The user file is only created on the next save_config().
        for legacy_path in _legacy_config_candidates():
            if os.path.exists(legacy_path):
                try:
                    current_config = _merge_loaded(
                        current_config, _read_json(legacy_path)
                    )
                except (json.JSONDecodeError, OSError):
                    continue
                break

    return current_config


def save_config(config_data) -> bool:
    """Persist ``config_data`` to the user configuration file.

    Only keys present in :data:`DEFAULT_CONFIG` are written, mirroring the
    behaviour of the former ``zemosaic_config`` module.
    """
    config_path = get_config_path()
    try:
        config_to_save = {}
        for key in DEFAULT_CONFIG:
            if key in config_data:
                config_to_save[key] = config_data[key]

        if not config_to_save and config_data:
            # No known key matched: save what we received rather than lose data.
            config_to_save = config_data
        elif not config_to_save and not config_data:
            return False  # Do not create an empty file.

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, indent=4, ensure_ascii=False)
        return True
    except OSError:
        return False


def get_astap_executable_path():
    return load_config().get("astap_executable_path", "")


def get_astap_data_directory_path():
    return load_config().get("astap_data_directory_path", "")


def get_astap_default_search_radius():
    return load_config().get(
        "astap_default_search_radius", DEFAULT_CONFIG["astap_default_search_radius"]
    )


def get_astap_default_downsample():
    return load_config().get(
        "astap_default_downsample", DEFAULT_CONFIG["astap_default_downsample"]
    )


def get_astap_default_sensitivity():
    return load_config().get(
        "astap_default_sensitivity", DEFAULT_CONFIG["astap_default_sensitivity"]
    )
