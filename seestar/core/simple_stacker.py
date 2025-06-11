import os
import numpy as np
from astropy.io import fits


def create_master_tile(
    seestar_stack_group_info,
    tile_id,
    output_temp_dir,
    stack_norm_method="none",
    stack_weight_method="none",
    stack_reject_algo="none",
    stack_kappa_low=3.0,
    stack_kappa_high=3.0,
    parsed_winsor_limits=(0.05, 0.05),
    stack_final_combine="mean",
    apply_radial_weight=False,
    radial_feather_fraction=0.8,
    radial_shape_power=2.0,
    min_radial_weight_floor=0.0,
    astap_exe_path_global="",
    astap_data_dir_global="",
    astap_search_radius_global=0.0,
    astap_downsample_global=0,
    astap_sensitivity_global=0,
    astap_timeout_seconds_global=0,
    progress_callback=None,
):
    """Simplified local replacement for ZeMosaic's ``create_master_tile``.

    It loads preprocessed image caches from ``seestar_stack_group_info`` and
    stacks them using a straightforward mean combine. Only a subset of the
    original parameters are currently honoured.
    """
    images = []
    for info in seestar_stack_group_info:
        path = info.get("path_preprocessed_cache")
        if not path or not os.path.exists(path):
            if progress_callback:
                progress_callback(f"⚠️ Cache manquant pour {path}", None)
            continue
        try:
            img = np.load(path)
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            images.append(img)
        except Exception as exc:
            if progress_callback:
                progress_callback(f"⚠️ Lecture échouée {path}: {exc}", None)
    if not images:
        if progress_callback:
            progress_callback("❌ Aucune image valide pour create_master_tile", None)
        return None, None

    data_stack = np.stack(images, axis=0)
    stacked = np.nanmean(data_stack, axis=0).astype(np.float32)

    hdr = fits.Header()
    hdr["NIMAGES"] = (len(images), "Images combined")

    os.makedirs(output_temp_dir, exist_ok=True)
    out_path = os.path.join(output_temp_dir, f"master_tile_{tile_id:03d}.fits")
    fits.writeto(out_path, np.moveaxis(stacked, -1, 0), hdr, overwrite=True)

    return out_path, None
