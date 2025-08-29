_missing = object()

try:
    from reproject import reproject_interp as _reproject_interp
except Exception:  # pragma: no cover - fallback when reproject missing
    def _missing_function(*_args, **_kwargs):
        raise ImportError(
            "The 'reproject' package is required for this functionality. "
            "Please install it with 'pip install reproject'."
        )

    _reproject_interp = _missing_function
    _missing = _reproject_interp

try:  # Prefer the reference implementation when available
    from reproject.mosaicking import reproject_and_coadd as _astropy_reproject_and_coadd
except Exception:  # pragma: no cover - gracefully handle absence of reproject
    _astropy_reproject_and_coadd = None

from astropy.wcs import WCS
from astropy.io import fits
import logging
import os
import gc
import tempfile

from seestar.core.image_processing import sanitize_header_for_wcs

logger = logging.getLogger(__name__)
import numpy as np


def _open_fits_safely(path):
    """Open a FITS file avoiding memmap scaling issues.

    The file is first inspected with ``memmap=True``. If scaling keywords
    (``BZERO``, ``BSCALE`` or ``BLANK``) are present the file is reopened with
    ``memmap=False`` and ``do_not_scale_image_data=False`` so that astropy can
    safely apply the required scaling. Otherwise a memory-mapped array is
    returned.
    """

    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        hdr = hdul[0].header.copy()
        needs_scaling = any(k in hdr for k in ("BZERO", "BSCALE", "BLANK"))
        if not needs_scaling:
            return hdul[0].data, hdr

    with fits.open(path, memmap=False, do_not_scale_image_data=False) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header.copy()
    return data, hdr


def _ensure_2d(img: np.ndarray) -> np.ndarray:
    """Force ``img`` to 2D, averaging the last axis when 3D."""

    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return img.mean(axis=-1)
    raise ValueError(f"Expected 2D image for reprojection, got shape {img.shape}")


def _to_chw(data: np.ndarray) -> np.ndarray:
    """Normalize ``data`` to ``(C, H, W)`` form.

    Handles grayscale ``(H, W)``, ``(C, H, W)`` and ``(H, W, C)`` inputs.
    Unknown layouts fall back to a single-channel view.
    """

    if data.ndim == 2:
        return data[None, ...]
    if data.shape[0] in (3, 4):
        return data
    if data.shape[-1] in (3, 4):
        return np.transpose(data, (2, 0, 1))
    return data[None, ...]


def reproject_and_coadd(
    input_data,
    output_projection,
    shape_out,
    input_weights=None,
    reproject_function=None,
    combine_function="mean",
    match_background=True,
    **kwargs,
):
    """Reproject all images and combine them on a common grid.

    Parameters
    ----------
    input_data : list of ``(array, WCS)``
        Sequence of image arrays with their associated input WCS objects.
    output_projection : astropy.wcs.WCS or FITS header
        Target projection defining the output grid.
    shape_out : tuple
        Desired output shape ``(H, W)``.
    input_weights : list of ndarray, optional
        Weight maps matching ``input_data`` shapes. If provided, they are
        reprojected and used when accumulating signal and coverage.
    reproject_function : callable, optional
        Function used to perform the reprojection. Defaults to
        :func:`reproject.reproject_interp`.

    Returns
    -------
    tuple
        ``(stacked, coverage)`` both ``np.ndarray`` with ``shape_out``.
    """

    if reproject_function is None:
        reproject_function = _reproject_interp

    if reproject_function is _missing:
        # reproject not available
        _reproject_interp()

    ref_wcs = WCS(output_projection) if not isinstance(output_projection, WCS) else output_projection
    shape_out = tuple(int(round(x)) for x in shape_out)


    weights_iter = input_weights if input_weights is not None else [None] * len(input_data)
    filtered_pairs = []
    filtered_weights = []
    for (img, wcs_in), weight in zip(input_data, weights_iter):
        wcs_obj = WCS(wcs_in) if not isinstance(wcs_in, WCS) else wcs_in
        if ref_wcs.has_celestial and not getattr(wcs_obj, "has_celestial", False):
            logger.warning("Skipping input without celestial WCS")
            continue
        filtered_pairs.append((img, wcs_obj))
        filtered_weights.append(weight)

    if not filtered_pairs:
        raise ValueError("No compatible input WCS for reprojection")

    use_astropy = _astropy_reproject_and_coadd is not None

    if use_astropy:
        mem_threshold = float(os.environ.get("REPROJECT_MEM_THRESHOLD_GB", "8"))
        mem_required = np.prod(shape_out) * 2 * 8 / 1024**3
        if mem_required > mem_threshold:
            logger.info(
                "Disabling astropy reproject_and_coadd (%.1f GiB required > %.1f GiB)",
                mem_required,
                mem_threshold,
            )
            use_astropy = False

    if use_astropy:
        # Use the reference implementation when possible but gracefully
        # fall back to the local implementation if it fails (e.g. due to
        # WCS incompatibilities). Older versions of ``reproject`` may
        # raise different exception types depending on the failure so we
        # simply catch ``Exception`` and only re-raise if it doesn't look
        # like a projection mismatch.
        try:
            return _astropy_reproject_and_coadd(
                filtered_pairs,
                output_projection=ref_wcs,
                shape_out=shape_out,
                input_weights=filtered_weights if input_weights is not None else None,
                reproject_function=reproject_function,
                combine_function=combine_function,
                match_background=match_background,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - depends on reproject version
            msg = str(exc)
            if "different number of world coordinates" not in msg.lower() and "output" not in msg.lower():
                raise

    sum_image = np.zeros(shape_out, dtype=np.float64)
    cov_image = np.zeros(shape_out, dtype=np.float64)


    for (img, wcs_in), weight in zip(filtered_pairs, filtered_weights):
        proj_img, footprint = reproject_function(
            (img, wcs_in), output_projection=ref_wcs, shape_out=shape_out, **kwargs
        )

        weight_proj = footprint
        if weight is not None:
            # [B1-COADD-FIX] accept scalar or 2D weights only
            if np.isscalar(weight):
                weight_proj = footprint * float(weight)
            else:
                weight = np.asarray(weight)
                if weight.shape != img.shape[:2]:
                    raise ValueError("weight shape mismatch")
                w_reproj, w_fp = reproject_function(
                    (weight, wcs_in), output_projection=ref_wcs, shape_out=shape_out, **kwargs
                )
                weight_proj = w_reproj * w_fp

        sum_image += proj_img * weight_proj
        cov_image += weight_proj

    final = np.full(shape_out, np.nan, dtype=np.float64)
    valid = cov_image > 0
    final[valid] = sum_image[valid] / cov_image[valid]

    return final.astype(np.float32), cov_image.astype(np.float32)


def reproject_and_coadd_from_paths(
    paths,
    output_projection=None,
    shape_out=None,
    **kwargs,
):
    """Load FITS files from ``paths`` then call :func:`reproject_and_coadd`.

    Parameters
    ----------
    paths : iterable of str
        FITS file paths containing valid WCS information.
    output_projection : astropy.wcs.WCS or FITS header, optional
        Target projection. When ``None`` the WCS of the first file is used.
    shape_out : tuple, optional
        Output shape ``(H, W)``. When ``None`` it is derived from the first
        image.
    **kwargs : dict
        Forwarded to :func:`reproject_and_coadd`.
    """

    pairs = []
    headers = []
    # [B1-COADD-FIX] collect meta information for global grid computation
    for fp in paths:
        try:
            with fits.open(fp, memmap=False) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float32)
                hdr = hdul[0].header
        except Exception:
            logger.warning("Skipping invalid FITS '%s' for reprojection", fp, exc_info=True)
            continue
        sanitize_header_for_wcs(hdr)
        wcs = WCS(hdr, naxis=2)
        if wcs.pixel_shape is None:
            h, w = data.shape[:2]
            wcs.pixel_shape = (w, h)
        if data.ndim == 3 and data.shape[0] in (1, 3) and data.shape[-1] != data.shape[0]:
            data = np.moveaxis(data, 0, -1)
        pairs.append((data, wcs))
        headers.append(hdr)
        logger.debug(
            "[B1-COADD-FIX] input=%s ndim=%d shape=%s", fp, data.ndim, data.shape
        )

    if not pairs:
        raise RuntimeError("Reproject requested but no aligned FITS were produced.")

    if output_projection is None or shape_out is None:
        try:  # pragma: no cover - depends on reproject availability
            from reproject.mosaicking import find_optimal_celestial_wcs

            out_wcs, out_shape = find_optimal_celestial_wcs(headers)
            logger.info("[B1-COADD-FIX] shape_out_global=%s", out_shape)
        except Exception:  # fall back to first image
            out_wcs = pairs[0][1]
            out_shape = pairs[0][0].shape[:2]
        if output_projection is None:
            output_projection = out_wcs
        if shape_out is None:
            shape_out = out_shape

    shape_out = tuple(int(round(x)) for x in shape_out)
    # [B1-COADD-FIX] sanity check on the final grid size
    if shape_out[0] < 100 or shape_out[1] < 100:
        raise ValueError(f"shape_out too small: {shape_out}")

    # Reproject per channel to avoid passing 3D arrays to reproject
    first = pairs[0][0]
    if first.ndim == 3:
        channels = []
        cov_out = None
        C = first.shape[-1]
        for c in range(C):
            ch_pairs = [(img[:, :, c], wcs) for img, wcs in pairs]
            sci, cov = reproject_and_coadd(
                ch_pairs, output_projection, shape_out, **kwargs
            )
            channels.append(sci)
            if cov_out is None:
                cov_out = cov
        result = np.stack(channels, axis=-1)
        return result, cov_out

    return reproject_and_coadd(pairs, output_projection, shape_out, **kwargs)


def streaming_reproject_and_coadd(
    paths,
    reference_path=None,
    output_path=None,
    tile_size=1024,
    dtype_out=np.float32,
    memmap_dir=None,
    keep_intermediates=False,
    match_background=True,
    reproject_function=None,
):
    """Streamingly reproject ``paths`` and coadd them on disk.

    Parameters
    ----------
    paths : list of str
        Paths to aligned FITS files.
    reference_path : str, optional
        FITS file providing the target WCS. Defaults to first path.
    output_path : str, optional
        Destination FITS file. When ``None`` the function returns ``False``.
    tile_size : int, optional
        Height/width of processing tiles.
    dtype_out : numpy dtype, optional
        Output dtype for the final stack.
    memmap_dir : str, optional
        Directory where temporary memmaps are stored.
    keep_intermediates : bool, optional
        If ``True`` temporary memmaps are preserved.
    match_background : bool, optional
        Currently unused placeholder for API compatibility.
    reproject_function : callable, optional
        Reprojection function, defaults to :func:`reproject.reproject_interp`.
    """

    if reproject_function is None:
        reproject_function = _reproject_interp

    if not paths:
        return False

    ref_fp = reference_path or paths[0]
    try:
        # Read raw reference data to preserve channels for C_out
        ref_data_raw, ref_hdr = _open_fits_safely(ref_fp)
        # Determine output 2D shape only (do not collapse channels for C_out)
        shape_out = _ensure_2d(ref_data_raw).shape[:2]
    except Exception:
        ref_data_raw = None
        ref_hdr = fits.Header()
        shape_out = None

    sanitize_header_for_wcs(ref_hdr)
    ref_wcs = WCS(ref_hdr, naxis=2)

    if shape_out is None:
        if ref_wcs.pixel_shape is not None and ref_wcs.array_shape is not None:
            shape_out = tuple(ref_wcs.array_shape)
        else:
            img_tmp, _ = _open_fits_safely(ref_fp)
            img_tmp = _ensure_2d(img_tmp)
            shape_out = img_tmp.shape[:2]

    if ref_wcs.pixel_shape is None or ref_wcs.array_shape is None:
        ref_wcs.pixel_shape = (shape_out[1], shape_out[0])
        ref_wcs.array_shape = shape_out

    # Compute C_out from raw reference data (correctly handles RGB vs mono)
    chw = _to_chw(ref_data_raw) if ref_data_raw is not None else None
    C_out = chw.shape[0] if chw is not None else 1

    if memmap_dir is None:
        memmap_dir = tempfile.mkdtemp(prefix="stream_reproject_")
    os.makedirs(memmap_dir, exist_ok=True)
    sum_path = os.path.join(memmap_dir, "sum_map.memmap")
    wht_path = os.path.join(memmap_dir, "wht_map.memmap")
    if C_out == 1:
        sum_map = np.memmap(sum_path, dtype=np.float64, mode="w+", shape=shape_out)
        wht_map = np.memmap(wht_path, dtype=np.float32, mode="w+", shape=shape_out)
    else:
        sum_map = np.memmap(sum_path, dtype=np.float64, mode="w+", shape=(C_out, *shape_out))
        wht_map = np.memmap(wht_path, dtype=np.float32, mode="w+", shape=(C_out, *shape_out))
    sum_map[:] = 0.0
    wht_map[:] = 0.0

    tile_h = tile_w = int(tile_size)

    for y0 in range(0, shape_out[0], tile_h):
        y1 = min(y0 + tile_h, shape_out[0])
        for x0 in range(0, shape_out[1], tile_w):
            x1 = min(x0 + tile_w, shape_out[1])
            sub_wcs = ref_wcs.slice((slice(y0, y1), slice(x0, x1)))
            for fp in paths:
                try:
                    data, hdr = _open_fits_safely(fp)
                    # Keep raw dimensionality; _to_chw will normalize to (C, H, W)
                except Exception:
                    logger.warning("Skipping invalid FITS '%s' for streaming reprojection", fp)
                    continue
                sanitize_header_for_wcs(hdr)
                in_wcs = WCS(hdr, naxis=2)
                chw_in = _to_chw(data)
                if chw_in.shape[0] != C_out:
                    if chw_in.shape[0] == 1 and C_out > 1:
                        # broadcast grayscale -> RGB-like
                        chw_in = np.repeat(chw_in, C_out, axis=0)
                    else:
                        logger.warning("Channel mismatch for '%s'", fp)
                        continue
                for c in range(C_out):
                    try:
                        arr, footprint = reproject_function(
                            (chw_in[c], in_wcs),
                            output_projection=sub_wcs,
                            shape_out=(y1 - y0, x1 - x0),
                            return_footprint=True,
                        )
                    except Exception:
                        logger.warning("Reprojection failed for '%s' (channel %d)", fp, c, exc_info=True)
                        continue
                    arr = arr.astype(dtype_out, copy=False)
                    footprint = footprint.astype(np.float32, copy=False)
                    if C_out == 1:
                        sum_map[y0:y1, x0:x1] += arr * footprint
                        wht_map[y0:y1, x0:x1] += footprint
                    else:
                        sum_map[c, y0:y1, x0:x1] += arr * footprint
                        wht_map[c, y0:y1, x0:x1] += footprint
                    del arr, footprint
                gc.collect()

    final_path = os.path.join(memmap_dir, "final.memmap")
    if C_out == 1:
        final_map = np.memmap(final_path, dtype=dtype_out, mode="w+", shape=shape_out)
    else:
        final_map = np.memmap(final_path, dtype=dtype_out, mode="w+", shape=(C_out, *shape_out))
    eps = np.finfo(np.float32).eps
    if C_out == 1:
        for y0 in range(0, shape_out[0], tile_h):
            y1 = min(y0 + tile_h, shape_out[0])
            for x0 in range(0, shape_out[1], tile_w):
                s = sum_map[y0:y1, x0:x1]
                w = wht_map[y0:y1, x0:x1]
                final_map[y0:y1, x0:x1] = np.where(w > 0, s / np.maximum(w, eps), np.nan)
    else:
        for c in range(C_out):
            for y0 in range(0, shape_out[0], tile_h):
                y1 = min(y0 + tile_h, shape_out[0])
                for x0 in range(0, shape_out[1], tile_w):
                    s = sum_map[c, y0:y1, x0:x1]
                    w = wht_map[c, y0:y1, x0:x1]
                    final_map[c, y0:y1, x0:x1] = np.where(
                        w > 0, s / np.maximum(w, eps), np.nan
                    )

    if output_path is not None:
        hdr_out = ref_wcs.to_header(relax=True)
        for key in ("BSCALE", "BZERO"):
            if key in ref_hdr:
                hdr_out[key] = ref_hdr[key]
        if C_out > 1:
            hdr_out["NAXIS"] = 3
            hdr_out["NAXIS1"] = shape_out[1]
            hdr_out["NAXIS2"] = shape_out[0]
            hdr_out["NAXIS3"] = C_out
            hdr_out["CTYPE3"] = hdr_out.get("CTYPE3", "RGB")
        fits.PrimaryHDU(data=final_map, header=hdr_out).writeto(
            output_path, overwrite=True
        )

    sum_map.flush(); wht_map.flush(); final_map.flush()
    if not keep_intermediates:
        try:
            os.remove(sum_path)
            os.remove(wht_path)
            os.remove(final_path)
            os.rmdir(memmap_dir)
        except Exception:
            pass
    return True


reproject_interp = _reproject_interp

__all__ = [
    "reproject_and_coadd",
    "reproject_interp",
    "reproject_and_coadd_from_paths",
    "streaming_reproject_and_coadd",
]
