"""Helper functions for incremental reprojection stacking."""

import numpy as np
import logging
from astropy.wcs import WCS
import importlib.util

_cupy_available = importlib.util.find_spec("cupy") is not None

from .reprojection import reproject_to_reference_wcs

logger = logging.getLogger(__name__)


def initialize_master(
    batch_img: np.ndarray,
    batch_cov: np.ndarray,
    batch_wcs: WCS,
    ref_wcs: WCS,
    use_gpu: bool | None = None,
):
    """Return initial weighted sum and coverage map.

    Parameters
    ----------
    batch_img : np.ndarray
        Stacked image from the first batch.
    batch_cov : np.ndarray
        Coverage/weight map for the batch.
    batch_wcs : astropy.wcs.WCS
        WCS of the first batch.
    ref_wcs : astropy.wcs.WCS
        Reference WCS used for later reprojection.

    The first batch is reprojected so that all subsequent batches
    accumulate on the same grid.
    """
    if use_gpu is None:
        use_gpu = _cupy_available

    xp = np
    cp = None
    if use_gpu and _cupy_available:
        import cupy as cp  # type: ignore
        xp = cp

    if batch_img is None or batch_cov is None:
        raise ValueError("batch_img and batch_cov are required")
    batch_img_f = xp.asarray(batch_img, dtype=xp.float32)
    batch_cov_f = xp.asarray(batch_cov, dtype=xp.float32)

    target_shape = (
        (ref_wcs.pixel_shape[1], ref_wcs.pixel_shape[0])
        if ref_wcs.pixel_shape is not None
        else batch_img_f.shape[:2]
    )

    reproj_img = reproject_to_reference_wcs(
        cp.asnumpy(batch_img_f) if cp is not None else batch_img_f,
        batch_wcs,
        ref_wcs,
        target_shape,
    )
    reproj_cov = reproject_to_reference_wcs(
        cp.asnumpy(batch_cov_f) if cp is not None else batch_cov_f,
        batch_wcs,
        ref_wcs,
        target_shape,
    )
    reproj_img = xp.asarray(reproj_img, dtype=xp.float32)
    reproj_cov = xp.asarray(reproj_cov, dtype=xp.float32)

    if reproj_img.ndim == 3:
        master_sum = reproj_img * reproj_cov[..., None]
    else:
        master_sum = reproj_img * reproj_cov

    master_cov = reproj_cov

    master_sum = xp.asarray(master_sum, dtype=xp.float32)
    master_cov = xp.asarray(master_cov, dtype=xp.float32)

    if cp is not None:
        master_sum = cp.asnumpy(master_sum)
        master_cov = cp.asnumpy(master_cov)

    return master_sum.astype(np.float32), master_cov.astype(np.float32)


def reproject_and_combine(
    master_sum: np.ndarray,
    master_cov: np.ndarray,
    batch_img: np.ndarray,
    batch_cov: np.ndarray,
    batch_wcs: WCS,
    ref_wcs: WCS,
    use_gpu: bool | None = None,
):
    """Reproject ``batch_img`` to ``ref_wcs`` and accumulate its weighted signal."""

    if batch_wcs is None or ref_wcs is None:
        return master_sum, master_cov


    if ref_wcs.pixel_shape is None:
        raise ValueError("ref_wcs.pixel_shape is required for reprojection")
    target_shape = (ref_wcs.pixel_shape[1], ref_wcs.pixel_shape[0])


    if use_gpu is None:
        use_gpu = _cupy_available

    xp = np
    cp = None
    if use_gpu and _cupy_available:
        import cupy as cp  # type: ignore
        xp = cp

    reproj_img = reproject_to_reference_wcs(batch_img, batch_wcs, ref_wcs, target_shape)
    reproj_cov = reproject_to_reference_wcs(batch_cov, batch_wcs, ref_wcs, target_shape)

    if reproj_img.shape[:2] != target_shape:
        # Fallback in case pixel_shape orientation was wrong
        target_shape = target_shape[::-1]
        reproj_img = reproject_to_reference_wcs(batch_img, batch_wcs, ref_wcs, target_shape)
        reproj_cov = reproject_to_reference_wcs(batch_cov, batch_wcs, ref_wcs, target_shape)

    master_sum = xp.asarray(master_sum, dtype=xp.float32)
    master_cov = xp.asarray(master_cov, dtype=xp.float32)
    reproj_img = xp.asarray(reproj_img, dtype=xp.float32)
    reproj_cov = xp.asarray(reproj_cov, dtype=xp.float32)

    if master_sum.ndim == 3:
        master_sum += reproj_img * reproj_cov[..., None]
    else:
        master_sum += reproj_img * reproj_cov

    master_cov += reproj_cov

    master_sum = xp.nan_to_num(master_sum, nan=0.0, posinf=0.0, neginf=0.0)
    master_cov = xp.nan_to_num(master_cov, nan=0.0, posinf=0.0, neginf=0.0)

    print(xp.nanmin(master_cov), xp.nanmax(master_cov))
    if cp is not None:
        master_sum = cp.asnumpy(master_sum)
        master_cov = cp.asnumpy(master_cov)
    return master_sum.astype(np.float32), master_cov.astype(np.float32)


def reproject_and_coadd_batch(
    image_array_list,
    header_list,
    target_wcs,
    target_shape_hw,
    combine_function="mean",
):
    """Reproject a batch of images and combine them with ``reproject_and_coadd``.

    Parameters
    ----------
    image_array_list : list of np.ndarray
        Images to reproject. Each array can be ``H x W`` or ``H x W x C``.
    header_list : list of ``astropy.io.fits.Header``
        FITS headers containing valid WCS for the corresponding images.
    target_wcs : astropy.wcs.WCS
        Output WCS grid to project onto.
    target_shape_hw : tuple
        Shape ``(H, W)`` of the output grid.
    combine_function : str, optional
        Combination method passed to ``reproject_and_coadd``.

    Returns
    -------
    tuple
        (combined_image, coverage_map) both ``np.ndarray`` in the target frame
        or ``(None, None)`` if reprojection failed.
    """

    from seestar.enhancement.reproject_utils import (
        reproject_and_coadd,
        reproject_interp,
    )

    if not image_array_list or not header_list:
        return None, None

    pairs = []
    weights = []
    for img, hdr in zip(image_array_list, header_list):
        if img is None or hdr is None:
            continue
        try:
            wcs = WCS(hdr, naxis=2)
            if not wcs.is_celestial:
                continue
            if wcs.pixel_shape is None:
                h, w = img.shape[:2]
                wcs.pixel_shape = (w, h)
        except Exception:
            continue
        pairs.append((img, wcs))
        weights.append(np.ones(img.shape[:2], dtype=np.float32))

    if not pairs:
        return None, None

    first_img = pairs[0][0]
    if first_img.ndim == 2:
        data_list = pairs
        cov_list = weights
        try:
            result, footprint = reproject_and_coadd(
                data_list,
                output_projection=target_wcs,
                shape_out=target_shape_hw,
                input_weights=cov_list,
                reproject_function=reproject_interp,
                combine_function=combine_function,
                match_background=True,
            )
        except Exception:
            logger.warning("reproject_and_coadd failed", exc_info=True)
            return None, None

        return result.astype(np.float32), footprint.astype(np.float32)

    # Colour images
    n_channels = first_img.shape[2]
    channel_arrays = [[] for _ in range(n_channels)]
    channel_weights = [[] for _ in range(n_channels)]
    for img, wcs in pairs:
        for c in range(n_channels):
            channel_arrays[c].append((img[:, :, c], wcs))
            channel_weights[c].append(np.ones(img.shape[:2], dtype=np.float32))

    final_channels = []
    final_cov = None
    for c in range(n_channels):
        try:
            res, cov = reproject_and_coadd(
                channel_arrays[c],
                output_projection=target_wcs,
                shape_out=target_shape_hw,
                input_weights=channel_weights[c],
                reproject_function=reproject_interp,
                combine_function=combine_function,
                match_background=True,
            )
        except Exception:
            logger.warning("reproject_and_coadd failed", exc_info=True)
            return None, None
        final_channels.append(res.astype(np.float32))
        if final_cov is None:
            final_cov = cov.astype(np.float32)

    combined = np.stack(final_channels, axis=-1)
    return combined, final_cov
