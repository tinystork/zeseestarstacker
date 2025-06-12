"""Helper functions for incremental reprojection stacking."""

import numpy as np
from astropy.wcs import WCS

from .reprojection import reproject_to_reference_wcs


def initialize_master(batch_img: np.ndarray, batch_cov: np.ndarray, ref_wcs: WCS):
    """Return initial weighted sum and coverage map.

    Parameters
    ----------
    batch_img : np.ndarray
        Stacked image from the first batch.
    batch_cov : np.ndarray
        Coverage/weight map for the batch.
    ref_wcs : astropy.wcs.WCS
        Reference WCS used for later reprojection.
    """
    if batch_img is None or batch_cov is None:
        raise ValueError("batch_img and batch_cov are required")
    batch_img_f = batch_img.astype(np.float32, copy=True)
    batch_cov_f = batch_cov.astype(np.float32, copy=True)

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
    """Reproject ``batch_img`` to ``ref_wcs`` and accumulate its weighted signal."""

    if batch_wcs is None or ref_wcs is None or ref_wcs.pixel_shape is None:
        return master_sum, master_cov

    target_shape = (ref_wcs.pixel_shape[1], ref_wcs.pixel_shape[0])
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
