"""Utility functions for WCS reprojection."""

from seestar.enhancement.reproject_utils import reproject_interp
import numpy as np
from astropy.wcs import WCS
import os
from multiprocessing import Pool


def reproject_to_reference_wcs(image_data, input_wcs, target_wcs, target_shape_hw):
    """Reproject ``image_data`` from ``input_wcs`` to ``target_wcs``.

    Parameters
    ----------
    image_data : np.ndarray
        Image array ``HxW`` or ``HxWxC`` (float32 recommended).
    input_wcs : astropy.wcs.WCS
        World Coordinate System describing ``image_data``.
    target_wcs : astropy.wcs.WCS
        Target WCS to reproject onto.
    target_shape_hw : tuple
        Shape ``(H, W)`` of the desired output.

    Returns
    -------
    np.ndarray
        Reprojected array with the same dtype as ``image_data`` and
        shape ``target_shape_hw`` (or ``target_shape_hw + (C,)`` if colour).

    Raises
    ------
    RuntimeError
        If the reprojection fails for any reason.
    """

    try:
        if not isinstance(input_wcs, WCS) or not isinstance(target_wcs, WCS):
            raise ValueError("input_wcs and target_wcs must be WCS instances")

        if image_data.ndim == 2:
            result, _ = reproject_interp((image_data, input_wcs),
                                          target_wcs,
                                          shape_out=tuple(target_shape_hw))
            return result.astype(image_data.dtype)

        if image_data.ndim == 3 and image_data.shape[2] in (3, 4):
            channels = []
            for ch in range(image_data.shape[2]):
                res, _ = reproject_interp(
                    (image_data[:, :, ch], input_wcs),
                    target_wcs,
                    shape_out=tuple(target_shape_hw),
                )
                channels.append(res.astype(image_data.dtype))
            return np.stack(channels, axis=2)

        raise ValueError("Unsupported image_data shape for reprojection")

    except Exception as exc:
        raise RuntimeError(f"Reprojection failed: {exc}")


# ----------------------------------------------------------------------
# Parallel WCS Reprojection Helper and Dispatcher
# ----------------------------------------------------------------------

def _reproject_single(args):
    """Helper for multiprocessing Pool: reproject a single image."""
    image, wcs_out, kwargs = args
    return reproject_interp(image, wcs_out, **kwargs)


def resolve_all_wcs(images, wcs_out, **kwargs):
    """Reproject a list of images onto ``wcs_out`` using multiple processes."""

    n_proc = max(1, os.cpu_count() // 2)
    tasks = [(img, wcs_out, kwargs) for img in images]

    with Pool(processes=n_proc) as pool:
        results = pool.map(_reproject_single, tasks)

    return results
