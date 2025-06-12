"""Helper functions for incremental reprojection stacking."""

import numpy as np
from astropy.wcs import WCS

from .reprojection import reproject_to_reference_wcs


def initialize_master(batch_img: np.ndarray, batch_cov: np.ndarray, ref_wcs: WCS):
    """Return initial master stack and coverage map.

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
    master_img = batch_img.astype(np.float32, copy=True)
    master_cov = batch_cov.astype(np.float32, copy=True)
    return master_img, master_cov


def reproject_and_combine(
    master_img: np.ndarray,
    master_cov: np.ndarray,
    batch_img: np.ndarray,
    batch_cov: np.ndarray,
    batch_wcs: WCS,
    ref_wcs: WCS,
):
    """Reproject ``batch_img`` to ``ref_wcs`` and combine with current master.

    The combination uses weighted averaging based on the coverage maps.
    """
    if batch_wcs is None or ref_wcs is None or ref_wcs.pixel_shape is None:
        return master_img, master_cov

    target_shape = (ref_wcs.pixel_shape[1], ref_wcs.pixel_shape[0])
    reproj_img = reproject_to_reference_wcs(batch_img, batch_wcs, ref_wcs, target_shape)
    reproj_cov = reproject_to_reference_wcs(batch_cov, batch_wcs, ref_wcs, target_shape)

    master_img = master_img.astype(np.float32, copy=False)
    master_cov = master_cov.astype(np.float32, copy=False)
    reproj_img = reproj_img.astype(np.float32, copy=False)
    reproj_cov = reproj_cov.astype(np.float32, copy=False)

    weight_total = master_cov + reproj_cov
    weight_total_safe = np.maximum(weight_total, 1e-9)

    if master_img.ndim == 3:
        master_img = (
            master_img * master_cov[..., None] + reproj_img * reproj_cov[..., None]
        ) / weight_total_safe[..., None]
    else:
        master_img = (master_img * master_cov + reproj_img * reproj_cov) / weight_total_safe

    master_cov = weight_total
    master_img = np.nan_to_num(master_img, nan=0.0, posinf=0.0, neginf=0.0)
    master_cov = np.nan_to_num(master_cov, nan=0.0, posinf=0.0, neginf=0.0)

    return master_img.astype(np.float32), master_cov.astype(np.float32)
