"""Image normalisation utilities duplicated from ZeMosaic."""
import numpy as np


def _normalize_images_linear_fit(image_list, reference_index=0, low_percentile=25.0, high_percentile=90.0):
    if not image_list or not (0 <= reference_index < len(image_list)):
        return [img.copy() if img is not None else None for img in image_list]
    ref = image_list[reference_index]
    if ref is None:
        return [img.copy() if img is not None else None for img in image_list]
    ref = ref.astype(np.float32, copy=False)
    is_color = ref.ndim == 3 and ref.shape[-1] == 3
    ref_low = np.nanpercentile(ref, low_percentile, axis=(0, 1) if is_color else None)
    ref_high = np.nanpercentile(ref, high_percentile, axis=(0, 1) if is_color else None)
    normalized = []
    for i, img in enumerate(image_list):
        if img is None:
            normalized.append(None)
            continue
        data = img.astype(np.float32, copy=False)
        if i == reference_index:
            normalized.append(data)
            continue
        low = np.nanpercentile(data, low_percentile, axis=(0, 1) if is_color else None)
        high = np.nanpercentile(data, high_percentile, axis=(0, 1) if is_color else None)
        delta_src = high - low
        delta_ref = ref_high - ref_low
        a = np.where(delta_src > 1e-5, delta_ref / np.maximum(delta_src, 1e-9), 1.0)
        b = ref_low - a * low
        data = a * data + b
        normalized.append(data)
    return normalized


def _normalize_images_sky_mean(image_list, reference_index=0, sky_percentile=25.0):
    if not image_list or not (0 <= reference_index < len(image_list)):
        return [img.copy() if img is not None else None for img in image_list]
    ref = image_list[reference_index]
    if ref is None:
        return [img.copy() if img is not None else None for img in image_list]
    ref = ref.astype(np.float32, copy=False)
    if ref.ndim == 3 and ref.shape[-1] == 3:
        ref_lum = 0.299 * ref[..., 0] + 0.587 * ref[..., 1] + 0.114 * ref[..., 2]
    else:
        ref_lum = ref
    ref_sky = np.nanpercentile(ref_lum, sky_percentile)
    normalized = []
    for i, img in enumerate(image_list):
        if img is None:
            normalized.append(None)
            continue
        data = img.astype(np.float32, copy=False)
        # Ensure the array is writeable for in-place adjustment
        if not data.flags.writeable:
            data = data.copy()
        if i == reference_index:
            normalized.append(data)
            continue
        if data.ndim == 3 and data.shape[-1] == 3:
            lum = 0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]
        else:
            lum = data
        curr_sky = np.nanpercentile(lum, sky_percentile)
        data += ref_sky - curr_sky
        normalized.append(data)
    return normalized
