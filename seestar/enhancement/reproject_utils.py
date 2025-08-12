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

    if not pairs:
        raise RuntimeError("Reproject requested but no aligned FITS were produced.")

    if output_projection is None:
        output_projection = pairs[0][1]
    if shape_out is None:
        shape_out = pairs[0][0].shape[:2]

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
        ref_hdr = fits.getheader(ref_fp, memmap=False)
    except Exception:
        ref_hdr = fits.Header()
    sanitize_header_for_wcs(ref_hdr)
    ref_wcs = WCS(ref_hdr, naxis=2)

    if ref_wcs.pixel_shape is None or ref_wcs.array_shape is None:
        with fits.open(ref_fp, memmap=True) as hdul:
            shape_out = hdul[0].data.shape[:2]
        ref_wcs.pixel_shape = (shape_out[1], shape_out[0])
        ref_wcs.array_shape = shape_out
    else:
        shape_out = tuple(ref_wcs.array_shape)

    if memmap_dir is None:
        memmap_dir = tempfile.mkdtemp(prefix="stream_reproject_")
    os.makedirs(memmap_dir, exist_ok=True)
    sum_path = os.path.join(memmap_dir, "sum_map.memmap")
    wht_path = os.path.join(memmap_dir, "wht_map.memmap")
    sum_map = np.memmap(sum_path, dtype=np.float64, mode="w+", shape=shape_out)
    wht_map = np.memmap(wht_path, dtype=np.float32, mode="w+", shape=shape_out)
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
                    with fits.open(fp, memmap=True) as hdul:
                        data = hdul[0].data
                        hdr = hdul[0].header
                except Exception:
                    logger.warning("Skipping invalid FITS '%s' for streaming reprojection", fp)
                    continue
                sanitize_header_for_wcs(hdr)
                in_wcs = WCS(hdr, naxis=2)
                try:
                    arr, footprint = reproject_function(
                        (data, in_wcs),
                        output_projection=sub_wcs,
                        shape_out=(y1 - y0, x1 - x0),
                        return_footprint=True,
                    )
                except Exception:
                    logger.warning("Reprojection failed for '%s'", fp, exc_info=True)
                    continue
                arr = arr.astype(dtype_out, copy=False)
                footprint = footprint.astype(np.float32, copy=False)
                sum_map[y0:y1, x0:x1] += arr * footprint
                wht_map[y0:y1, x0:x1] += footprint
                del arr, footprint
                gc.collect()

    final_path = os.path.join(memmap_dir, "final.memmap")
    final_map = np.memmap(final_path, dtype=dtype_out, mode="w+", shape=shape_out)
    eps = np.finfo(np.float32).eps
    for y0 in range(0, shape_out[0], tile_h):
        y1 = min(y0 + tile_h, shape_out[0])
        for x0 in range(0, shape_out[1], tile_w):
            s = sum_map[y0:y1, x0:x1]
            w = wht_map[y0:y1, x0:x1]
            final_map[y0:y1, x0:x1] = np.where(w > 0, s / np.maximum(w, eps), np.nan)

    if output_path is not None:
        fits.PrimaryHDU(data=final_map, header=ref_wcs.to_header(relax=True)).writeto(
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
