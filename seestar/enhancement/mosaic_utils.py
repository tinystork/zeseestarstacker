import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .reproject_utils import reproject_and_coadd, reproject_interp
from .weight_utils import make_radial_weight_map
from zemosaic import zemosaic_utils
import inspect


def assemble_final_mosaic_with_reproject_coadd(
    master_tile_fits_with_wcs_list,
    final_output_wcs: WCS,
    final_output_shape_hw: tuple,
    match_bg: bool = True,
):
    """Assemble master tiles using ``reproject_and_coadd``.

    Parameters
    ----------
    master_tile_fits_with_wcs_list : list
        List of ``(path, WCS)`` tuples for stacked batches.
    final_output_wcs : astropy.wcs.WCS
        Target WCS of the mosaic.
    final_output_shape_hw : tuple
        Shape ``(H, W)`` of the final mosaic.
    match_bg : bool, optional
        Forwarded to ``reproject_and_coadd``.

    Returns
    -------
    tuple
        (mosaic_hwc, coverage_hw) both ``np.ndarray`` or ``(None, None)`` on
        failure.
    """

    if not master_tile_fits_with_wcs_list:
        return None, None
    h, w = map(int, final_output_shape_hw)
    try:
        w_wcs = int(getattr(final_output_wcs, "pixel_shape", (w, h))[0])
        h_wcs = int(getattr(final_output_wcs, "pixel_shape", (w, h))[1])
    except Exception:
        w_wcs = int(getattr(final_output_wcs.wcs, "naxis1", w)) if hasattr(final_output_wcs, "wcs") else w
        h_wcs = int(getattr(final_output_wcs.wcs, "naxis2", h)) if hasattr(final_output_wcs, "wcs") else h
    expected_hw = (h_wcs, w_wcs)
    if (h, w) != expected_hw:
        if (w, h) == expected_hw:
            final_output_shape_hw = expected_hw
            h, w = final_output_shape_hw
        else:
            return None, None

    output_header = (
        final_output_wcs.to_header()
        if hasattr(final_output_wcs, "to_header")
        else final_output_wcs
    )

    channel_data = [[] for _ in range(3)]
    channel_wht = [[] for _ in range(3)]
    wcs_list = []

    for path, wcs in master_tile_fits_with_wcs_list:
        try:
            with fits.open(path, memmap=False) as hdul:
                data = hdul[0].data.astype(np.float32)
        except Exception:
            continue

        if data.ndim == 3 and data.shape[0] in (1, 3):
            data = np.moveaxis(data, 0, -1)

        cov = np.ones(data.shape[:2], dtype=np.float32)
        cov *= make_radial_weight_map(*cov.shape)

        wcs_list.append(wcs)
        for ch in range(data.shape[2]):
            channel_data[ch].append(data[..., ch])
            channel_wht[ch].append(cov)

    mosaic_channels = []
    coverage = None
    for ch in range(3):
        try:
            kwargs = {}
            try:
                sig = inspect.signature(reproject_and_coadd)
                if "match_background" in sig.parameters:
                    kwargs["match_background"] = match_bg
                elif "match_bg" in sig.parameters:
                    kwargs["match_bg"] = match_bg
            except Exception:
                kwargs["match_background"] = match_bg

            sci, cov = zemosaic_utils.reproject_and_coadd_wrapper(
                data_list=channel_data[ch],
                wcs_list=wcs_list,
                shape_out=final_output_shape_hw,
                output_projection=output_header,
                use_gpu=False,
                cpu_func=reproject_and_coadd,
                reproject_function=reproject_interp,
                combine_function="mean",
                input_weights=channel_wht[ch],
                **kwargs,
            )
        except Exception:
            return None, None
        mosaic_channels.append(sci.astype(np.float32))
        if coverage is None:
            coverage = cov.astype(np.float32)

    mosaic = np.stack(mosaic_channels, axis=-1)
    return mosaic, coverage
