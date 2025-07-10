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
import logging

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

    if _astropy_reproject_and_coadd is not None:
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


reproject_interp = _reproject_interp

__all__ = ["reproject_and_coadd", "reproject_interp"]
