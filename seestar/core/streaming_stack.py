import os
import gc
import logging
import time
from typing import Sequence, Iterable

import numpy as np
from astropy.io import fits
import psutil
from concurrent.futures import ThreadPoolExecutor

from .stack_methods import (
    _stack_mean,
    _stack_median,
    _stack_kappa_sigma,
    _stack_winsorized_sigma,
    _stack_linear_fit_clip,
)

logger = logging.getLogger(__name__)

_PROC = psutil.Process(os.getpid())

def _log_mem(tag: str) -> None:
    try:
        rss = _PROC.memory_info().rss / (1024 * 1024)
        logger.debug("RAM [%s]: %.1f MB", tag, rss)
    except Exception:
        pass


def _auto_chunk_rows(num_images: int, img_shape: tuple[int, ...]) -> int:
    """Determine chunk height based on available RAM.

    Parameters
    ----------
    num_images : int
        Number of images stacked simultaneously.
    img_shape : tuple[int, ...]
        Shape of one image ``(H, W[, C])``.

    Returns
    -------
    int
        Recommended number of rows to process at once.
    """

    avail = psutil.virtual_memory().available
    h, w = img_shape[:2]
    c = 1 if len(img_shape) == 2 else img_shape[2]
    bytes_per_row = num_images * w * c * 4  # float32
    if bytes_per_row <= 0:
        return 1
    max_rows = max(1, int((avail * 0.2) // bytes_per_row))
    return min(max_rows, h)


def estimate_parallel_io_settings(
    num_files: int,
    shape_hw: tuple[int, int],
    max_ram_frac: float = 0.3,
    max_workers_cap: int = 8,
) -> tuple[int, int]:
    """Estimate ``chunk_rows`` and ``max_workers`` for parallel streaming."""
    import psutil as _psutil

    h, w = shape_hw
    available = _psutil.virtual_memory().available
    target_mem = available * max_ram_frac

    img_mem_per_row = w * 4  # float32 bytes per row
    max_chunk_rows = max(1, int(target_mem // (img_mem_per_row * num_files)))

    chunk_rows = min(256, max_chunk_rows)
    chunk_rows = max(8, chunk_rows)

    bytes_per_image_chunk = chunk_rows * w * 4
    max_possible_workers = int(target_mem // bytes_per_image_chunk)
    max_workers = min(num_files, max_possible_workers, max_workers_cap)

    return chunk_rows, max_workers


def stack_disk_streaming(
    file_list: Sequence[str],
    *,
    mode: str = "mean",
    weights: Iterable[float] | None = None,
    chunk_rows: int = 256,
    kappa: float = 3.0,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    winsor_limits: tuple[float, float] = (0.05, 0.05),
    apply_rewinsor: bool = True,
    parallel_io: bool = True,
) -> str:
    """Stack images from ``file_list`` using small row chunks.

    Parameters
    ----------
    file_list : Sequence[str]
        List of FITS file paths.
    mode : str, optional
        Stacking mode. One of ``mean``, ``median``, ``kappa-sigma``,
        ``winsorized-sigma`` or ``linear_fit_clip``.
    weights : Iterable[float], optional
        Optional weight scalar per image.
    chunk_rows : int, optional
        Number of rows to load at once.
    kappa, sigma_low, sigma_high, winsor_limits : float
        Parameters for rejection algorithms.
    apply_rewinsor : bool, optional
        See ``_stack_winsorized_sigma``.
    parallel_io : bool, optional
        Read chunks from disk in parallel when ``True``.

    Returns
    -------
    str
        Path to the stacked FITS file on disk.
    """
    if not file_list:
        raise ValueError("file_list is empty")

    first_path = file_list[0]
    if first_path.lower().endswith(".npy"):
        arr0 = np.load(first_path, mmap_mode="r")
        shape = arr0.shape
    else:
        with fits.open(first_path, memmap=True) as hdul0:
            shape = tuple(hdul0[0].data.shape)

    is_cf = (
        len(shape) == 3
        and shape[0] in (1, 3, 4)
        and shape[1] > 4
        and shape[2] > 4
    )
    if is_cf:
        C, H, W = shape
    else:
        H, W = shape[:2]
        C = 1 if len(shape) == 2 else shape[2]

    out_shape = (H, W) if C == 1 else (C, H, W)
    out_path = os.path.join(os.getcwd(), "stack_result.fits")
    out_hdu = fits.PrimaryHDU(np.zeros(out_shape, dtype=np.float32))
    fits.HDUList([out_hdu]).writeto(out_path, overwrite=True)

    weights = list(weights) if weights is not None else [1.0] * len(file_list)
    weights_arr = np.asarray(weights, dtype=np.float32)

    if chunk_rows <= 0:
        chunk_rows, max_workers = estimate_parallel_io_settings(
            len(file_list), (H, W), max_ram_frac=0.3
        )
    else:
        max_workers = min(4, os.cpu_count() or 1)

    logger.debug(
        "Streaming stack: %d files, chunk_rows=%d, workers=%d, shape=%s",
        len(file_list),
        chunk_rows,
        max_workers,
        (H, W, C) if C > 1 else (H, W),
    )

    start = time.perf_counter()
    with fits.open(out_path, mode="update", memmap=True) as out_hdul:
        for row_start in range(0, H, chunk_rows):
            row_end = min(row_start + chunk_rows, H)
            if parallel_io:
                def _read_chunk(path, row_start=row_start, row_end=row_end):
                    try:
                        if path.lower().endswith(".npy"):
                            arr = np.load(path, mmap_mode="r")
                        else:
                            with fits.open(path, memmap=True) as hdul:
                                arr = hdul[0].data
                        if is_cf:
                            sl = arr[:, row_start:row_end, :]
                        else:
                            sl = arr[row_start:row_end]
                            if C > 1:
                                sl = np.moveaxis(sl, -1, 0)
                            else:
                                sl = sl[np.newaxis, ...]
                        return sl.astype(np.float32, copy=False)
                    except Exception as e:
                        logger.warning(f"Failed to read chunk from {path}: {e}")
                        return None

                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    chunk_slices = list(pool.map(lambda p: _read_chunk(p), file_list))
                chunk_slices = [sl for sl in chunk_slices if sl is not None]
                arr_cf = np.stack(chunk_slices, axis=0)
                del chunk_slices
            else:
                chunk_slices = []
                for path in file_list:
                    if path.lower().endswith(".npy"):
                        arr = np.load(path, mmap_mode="r")
                    else:
                        with fits.open(path, memmap=True) as hdul:
                            arr = hdul[0].data
                    if is_cf:
                        sl = arr[:, row_start:row_end, :]
                    else:
                        sl = arr[row_start:row_end]
                        if C > 1:
                            sl = np.moveaxis(sl, -1, 0)
                        else:
                            sl = sl[np.newaxis, ...]
                    chunk_slices.append(sl.astype(np.float32, copy=False))
                arr_cf = np.stack(chunk_slices, axis=0)
                del chunk_slices

            if C == 1:
                arr = arr_cf[:, 0, :, :]
            else:
                arr = np.moveaxis(arr_cf, 1, -1)

            if mode == "median":
                chunk_result, _ = _stack_median(arr, weights_arr)
            elif mode == "mean":
                chunk_result, _ = _stack_mean(arr, weights_arr)
            elif mode == "kappa-sigma":
                chunk_result, _ = _stack_kappa_sigma(
                    arr, weights_arr, sigma_low=sigma_low, sigma_high=sigma_high
                )
            elif mode == "winsorized-sigma":
                chunk_result, _ = _stack_winsorized_sigma(
                    arr,
                    weights_arr,
                    kappa=kappa,
                    winsor_limits=winsor_limits,
                    apply_rewinsor=apply_rewinsor,
                )
            elif mode == "linear_fit_clip":
                chunk_result, _ = _stack_linear_fit_clip(arr, weights_arr)
            else:
                raise ValueError(f"Unknown stacking mode: {mode}")

            if C == 1:
                out_hdul[0].data[row_start:row_end] = chunk_result.astype(np.float32)
            else:
                out_hdul[0].data[:, row_start:row_end, :] = np.moveaxis(
                    chunk_result.astype(np.float32), -1, 0
                )
            out_hdul.flush()
            del arr, arr_cf, chunk_result
            gc.collect()
            _log_mem(f"chunk_{row_start}_{row_end}")

    gc.collect()
    _log_mem("stream_done")
    elapsed = time.perf_counter() - start
    logger.debug(
        "Streaming stack complete in %.2fs (%d temp files)",
        elapsed,
        len(file_list),
    )
    return out_path


# -----------------------------------------------------------------------------
# Reprojection streaming wrapper
# -----------------------------------------------------------------------------

from astropy.stats import sigma_clip
from astropy.wcs import WCS


def _subtract_sky_median(image, nsig: float = 3.0, maxiters: int = 5):
    """Subtract a robust sky median from ``image`` without mask warnings."""

    clipped = sigma_clip(image, sigma=nsig, maxiters=maxiters)
    med = np.nanmedian(clipped.filled(np.nan))
    med_val = 0.0 if not np.isfinite(med) else float(med)
    return image - med_val


def streaming_reproject_and_coadd(
    aligned_paths: Sequence[str],
    output_wcs: WCS | None = None,
    shape_out: tuple[int, int] | None = None,
    subtract_sky_median: bool = True,
    **kwargs,
):
    """Proxy to enhancement.streaming_reproject_and_coadd.

    This thin wrapper lives in ``core`` to avoid circular imports while
    offering a stable entry point for higher level modules.
    """

    from ..enhancement.reproject_utils import streaming_reproject_and_coadd as _impl

    return _impl(
        aligned_paths,
        output_wcs=output_wcs,
        shape_out=shape_out,
        subtract_sky_median=subtract_sky_median,
        **kwargs,
    )
