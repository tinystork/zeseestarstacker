"""Helper functions for incremental reprojection stacking."""

import numpy as np
from astropy.wcs import WCS

from .reprojection import reproject_to_reference_wcs


def initialize_master(
    batch_img: np.ndarray,
    batch_cov: np.ndarray,
    ref_wcs: WCS,
    batch_wcs: WCS | None = None,
):
    """Return initial weighted sum and coverage map.

    Parameters
    ----------
    batch_img : np.ndarray
        Stacked image from the first batch.
    batch_cov : np.ndarray
        Coverage/weight map for the batch.
    ref_wcs : astropy.wcs.WCS
        Reference WCS used for later reprojection.
    batch_wcs : astropy.wcs.WCS, optional
        WCS of ``batch_img`` for reprojection of the first batch.
    """
    if batch_img is None or batch_cov is None:
        raise ValueError("batch_img and batch_cov are required")
    batch_img_f = batch_img.astype(np.float32, copy=True)
    batch_cov_f = batch_cov.astype(np.float32, copy=True)

    if batch_wcs is not None and ref_wcs.pixel_shape is not None:
        target_shape = (ref_wcs.pixel_shape[1], ref_wcs.pixel_shape[0])
        reproj_img = reproject_to_reference_wcs(
            batch_img_f, batch_wcs, ref_wcs, target_shape
        )
        reproj_cov = reproject_to_reference_wcs(
            batch_cov_f, batch_wcs, ref_wcs, target_shape
        )
        if reproj_img.ndim == 3:
            master_sum = reproj_img * reproj_cov[..., None]
        else:
            master_sum = reproj_img * reproj_cov
        master_cov = reproj_cov
    else:
        if batch_img_f.ndim == 3:
            master_sum = batch_img_f * batch_cov_f[..., None]
        else:
            master_sum = batch_img_f * batch_cov_f
        master_cov = batch_cov_f

    return master_sum, master_cov


def reproject_and_combine(
    master_sum: np.ndarray,
    master_cov: np.ndarray,
    batch_img: np.ndarray,
    batch_cov: np.ndarray,
    batch_wcs: WCS,
    ref_wcs: WCS,
):
    """Reproject ``batch_img`` to ``ref_wcs`` and accumulate its weighted signal.

    Parameters
    ----------
    master_sum : np.ndarray
        Current accumulated weighted signal.
    master_cov : np.ndarray
        Current accumulated coverage map.
    batch_img : np.ndarray
        Image from the next batch to add.
    batch_cov : np.ndarray
        Coverage/weight map for ``batch_img``.
    batch_wcs : astropy.wcs.WCS
        WCS describing ``batch_img``.
    ref_wcs : astropy.wcs.WCS
        Reference WCS defining the target grid. ``ref_wcs.pixel_shape`` is used
        to compute the reprojection geometry.
    """

    if batch_wcs is None or ref_wcs is None:
        return master_sum, master_cov

    if ref_wcs.pixel_shape is not None:
        target_shape = (ref_wcs.pixel_shape[1], ref_wcs.pixel_shape[0])
    else:
        target_shape = master_sum.shape[:2]

    reproj_img = reproject_to_reference_wcs(batch_img, batch_wcs, ref_wcs, target_shape)
    reproj_cov = reproject_to_reference_wcs(batch_cov, batch_wcs, ref_wcs, target_shape)

    master_sum = master_sum.astype(np.float32, copy=False)
    master_cov = master_cov.astype(np.float32, copy=False)
    reproj_img = reproj_img.astype(np.float32, copy=False)
    reproj_cov = reproj_cov.astype(np.float32, copy=False)

    if master_sum.ndim == 3:
        master_sum += reproj_img * reproj_cov[..., None]
    else:
        master_sum += reproj_img * reproj_cov

    master_cov += reproj_cov

    master_sum = np.nan_to_num(master_sum, nan=0.0, posinf=0.0, neginf=0.0)
    master_cov = np.nan_to_num(master_cov, nan=0.0, posinf=0.0, neginf=0.0)

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
        result, footprint = reproject_and_coadd(
            data_list,
            output_projection=target_wcs,
            shape_out=target_shape_hw,
            input_weights=cov_list,
            reproject_function=reproject_interp,
            combine_function=combine_function,
            match_background=True,
        )
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
        res, cov = reproject_and_coadd(
            channel_arrays[c],
            output_projection=target_wcs,
            shape_out=target_shape_hw,
            input_weights=channel_weights[c],
            reproject_function=reproject_interp,
            combine_function=combine_function,
            match_background=True,
        )
        final_channels.append(res.astype(np.float32))
        if final_cov is None:
            final_cov = cov.astype(np.float32)

    combined = np.stack(final_channels, axis=-1)
    return combined, final_cov
