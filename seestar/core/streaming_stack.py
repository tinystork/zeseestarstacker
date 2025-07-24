import os
import gc
import logging
import time
from typing import Sequence, Iterable

import numpy as np
from astropy.io import fits
import psutil

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

    Returns
    -------
    str
        Path to the stacked FITS file on disk.
    """
    if not file_list:
        raise ValueError("file_list is empty")

    with fits.open(file_list[0], memmap=True) as hdul0:
        shape = tuple(hdul0[0].data.shape)
        dtype = hdul0[0].data.dtype
    H, W = shape[:2]
    C = 1 if len(shape) == 2 else shape[2]

    out_path = os.path.join(os.getcwd(), "stack_result.fits")
    out_hdu = fits.PrimaryHDU(np.zeros(shape, dtype=np.float32))
    fits.HDUList([out_hdu]).writeto(out_path, overwrite=True)

    weights = list(weights) if weights is not None else [1.0] * len(file_list)
    weights_arr = np.asarray(weights, dtype=np.float32)

    if chunk_rows <= 0:
        chunk_rows = _auto_chunk_rows(len(file_list), shape)

    logger.debug(
        "Streaming stack: %d files, chunk_rows=%d, shape=%s",
        len(file_list),
        chunk_rows,
        shape,
    )

    start = time.perf_counter()
    with fits.open(out_path, mode="update", memmap=True) as out_hdul:
        for row_start in range(0, H, chunk_rows):
            row_end = min(row_start + chunk_rows, H)
            chunk_slices = []
            for path in file_list:
                with fits.open(path, memmap=True) as hdul:
                    sl = hdul[0].data[row_start:row_end]
                    chunk_slices.append(sl.astype(np.float32, copy=False))
            arr = np.stack(chunk_slices, axis=0)
            del chunk_slices

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

            out_hdul[0].data[row_start:row_end] = chunk_result.astype(np.float32)
            out_hdul.flush()
            del arr, chunk_result
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
