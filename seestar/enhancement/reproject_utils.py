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

from astropy.wcs import WCS
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
    shape_out = tuple(shape_out)

    sum_image = np.zeros(shape_out, dtype=np.float64)
    cov_image = np.zeros(shape_out, dtype=np.float64)

    weights_iter = input_weights if input_weights is not None else [None] * len(input_data)

    for (img, wcs_in), weight in zip(input_data, weights_iter):
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


reproject_interp = _reproject_interp

__all__ = ["reproject_and_coadd", "reproject_interp"]
