import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .reproject_utils import reproject_and_coadd, reproject_interp

from zemosaic import zemosaic_utils
import inspect
import os
import shutil



def assemble_final_mosaic_with_reproject_coadd(
    master_tile_fits_with_wcs_list,
    final_output_wcs: WCS,
    final_output_shape_hw: tuple,
    match_bg: bool = True,
    weight_arrays=None,
    use_memmap: bool = False,
    memmap_dir: str | None = None,
    cleanup_memmap: bool = True,
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
    weight_arrays : list of ndarray, optional
        Optional per-tile weight maps passed to ``reproject_and_coadd``.
    use_memmap : bool, optional
        If ``True`` and supported by the underlying ``reproject`` version,
        intermediate arrays are memory-mapped to ``memmap_dir`` to reduce RAM
        usage.
    memmap_dir : str, optional
        Directory where memmap files are stored. Created if needed. When
        ``None``, no explicit directory is passed to ``reproject``.
    cleanup_memmap : bool, optional
        When ``True`` the memmap directory will be removed once stacking
        completes.

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


    data_all = []

    wcs_list = []

    for path, wcs in master_tile_fits_with_wcs_list:
        try:
            with fits.open(path, memmap=False) as hdul:
                data = hdul[0].data.astype(np.float32)
        except Exception:
            continue

        if data.ndim == 3 and data.shape[0] in (1, 3) and data.shape[-1] != data.shape[0]:
            data = np.moveaxis(data, 0, -1)
        if data.ndim == 2:
            data = data[..., np.newaxis]


        data_all.append(data)
        wcs_list.append(wcs)


    mosaic_channels = []
    coverage = None
    n_ch = data_all[0].shape[2] if data_all else 0

    header = (
        final_output_wcs.to_header()
        if hasattr(final_output_wcs, "to_header")
        else final_output_wcs
    )
    # ``reproject`` expects the output grid size to be present in the header
    # when a ``WCS`` instance is provided.  Explicitly set ``NAXIS1``/``NAXIS2``
    # so that the reference implementation and our NumPy fallback both
    # operate on the same shape regardless of ``final_output_wcs`` internals.
    header["NAXIS1"] = w
    header["NAXIS2"] = h

    if use_memmap:
        if memmap_dir is None:
            memmap_dir = os.path.join(os.getcwd(), "reproject_memmap")
        os.makedirs(memmap_dir, exist_ok=True)

    for ch in range(n_ch):
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

            data_list = [arr[..., ch] for arr in data_all]

            kwargs_local = dict(kwargs)
            if weight_arrays is not None:
                weights_ch = []
                for w in weight_arrays:
                    if w.ndim == 3:
                        if w.shape[0] == n_ch:
                            weights_ch.append(w[ch])
                            continue
                        if w.shape[-1] == n_ch:
                            weights_ch.append(w[..., ch])
                            continue
                        w = np.squeeze(w)
                    weights_ch.append(w)
                kwargs_local["input_weights"] = weights_ch
            try:
                sig = inspect.signature(reproject_and_coadd)
                if use_memmap:
                    if "use_memmap" in sig.parameters:
                        kwargs_local["use_memmap"] = True
                    elif "intermediate_memmap" in sig.parameters:
                        kwargs_local["intermediate_memmap"] = True
                    if "memmap_dir" in sig.parameters:
                        kwargs_local["memmap_dir"] = memmap_dir
                    if "cleanup_memmap" in sig.parameters:
                        kwargs_local["cleanup_memmap"] = False
            except Exception:
                if use_memmap and "memmap_dir" not in kwargs_local:
                    kwargs_local["memmap_dir"] = memmap_dir

            sci, cov = zemosaic_utils.reproject_and_coadd_wrapper(
                data_list=data_list,
                wcs_list=wcs_list,
                shape_out=final_output_shape_hw,

                output_projection=header,

                use_gpu=False,
                cpu_func=reproject_and_coadd,
                reproject_function=reproject_interp,
                combine_function="mean",

                **kwargs_local,
            )
        except Exception:
            return None, None
        mosaic_channels.append(sci.astype(np.float32))
        if coverage is None:
            coverage = cov.astype(np.float32)

    mosaic = np.stack(mosaic_channels, axis=-1)
    if use_memmap and cleanup_memmap and memmap_dir:
        try:
            shutil.rmtree(memmap_dir)
        except Exception:
            pass
    return mosaic, coverage
