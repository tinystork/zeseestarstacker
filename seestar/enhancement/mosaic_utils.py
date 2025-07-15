import logging
import os
import shutil
import inspect
from typing import Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from scipy.ndimage import binary_dilation

from .reproject_utils import reproject_and_coadd, reproject_interp

from zemosaic import zemosaic_utils

logger = logging.getLogger(__name__)

try:
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    _PHOTUTILS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    DAOStarFinder = None  # type: ignore
    sigma_clipped_stats = None  # type: ignore
    _PHOTUTILS_AVAILABLE = False


def detect_stars(img: np.ndarray) -> np.ndarray:
    """Return boolean mask of detected stars in ``img``.

    Parameters
    ----------
    img : np.ndarray
        2-D input image.

    Returns
    -------
    np.ndarray
        Boolean mask where ``True`` indicates stellar pixels.
    """

    if not _PHOTUTILS_AVAILABLE:
        raise ImportError("photutils is required for detect_stars")

    if img.ndim != 2:
        raise ValueError("detect_stars expects a 2-D array")

    mean, median, std = sigma_clipped_stats(img, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
    sources = daofind(img - median)
    mask = np.zeros_like(img, dtype=bool)
    if sources is not None and len(sources) > 0:
        y, x = sources["ycentroid"].astype(int), sources["xcentroid"].astype(int)
        y = np.clip(y, 0, img.shape[0] - 1)
        x = np.clip(x, 0, img.shape[1] - 1)
        mask[y, x] = True
    if mask.any():
        mask = binary_dilation(mask, iterations=2)
    return mask


def _background_metrics(img: np.ndarray, star_mask: np.ndarray) -> Tuple[float, float, float]:
    """Compute background statistics for ``img`` excluding stars."""

    if img.ndim != 2:
        raise ValueError("_background_metrics expects a 2-D array")

    if img.shape != star_mask.shape:
        raise ValueError("star_mask must match image shape")

    sample = img
    mask = star_mask
    if max(img.shape) > 2048:
        step = 4
        sample = img[::step, ::step]
        mask = star_mask[::step, ::step]

    background_pixels = sample[~mask]
    med = float(np.median(background_pixels)) if background_pixels.size else 0.0
    mad = float(np.median(np.abs(background_pixels - med))) if background_pixels.size else 0.0
    noise = mad * 1.4826

    y, x = np.indices(sample.shape)
    A = np.column_stack((x[~mask].ravel(), y[~mask].ravel(), np.ones(np.count_nonzero(~mask))))
    b = sample[~mask].ravel()
    try:
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    except Exception:
        plane = np.zeros_like(sample, dtype=float)
    residual = sample - plane
    gy, gx = np.gradient(residual)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad_rms = float(np.sqrt(np.mean(grad[~mask] ** 2))) if np.any(~mask) else 0.0
    return med, noise, grad_rms


def coverage_fraction(idx: int, footprints: np.ndarray) -> float:
    """Return fraction of well-covered pixels for image ``idx``."""

    if footprints.ndim != 3:
        raise ValueError("footprints must be 3-D array")
    fp = footprints[idx] > 0
    overlap = (footprints > 0).sum(axis=0)
    good = fp & (overlap >= 2)
    return float(np.count_nonzero(good)) / float(fp.size)



def assemble_final_mosaic_with_reproject_coadd(
    master_tile_fits_with_wcs_list,
    final_output_wcs: WCS,
    final_output_shape_hw: tuple,
    match_bg: bool = True,
    weight_arrays=None,
    use_memmap: bool = False,
    memmap_dir: str | None = None,
    cleanup_memmap: bool = True,
    auto_bg_reference: bool = True,
    force_reference_index: int | None = None,
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
    auto_bg_reference : bool, optional
        When ``True`` and background matching is enabled, automatically select
        the best reference image for background normalization.
    force_reference_index : int, optional
        If provided, override automatic selection and use this index as the
        background reference.

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

    header = final_output_wcs.to_header(relax=True)

    if use_memmap:
        if memmap_dir is None:
            memmap_dir = os.path.join(os.getcwd(), "reproject_memmap")
        os.makedirs(memmap_dir, exist_ok=True)

    best_ref = None
    if auto_bg_reference and match_bg:
        footprints = []
        for arr, wcs in zip(data_all, wcs_list):
            try:
                _, fp = reproject_interp(
                    (arr[..., 0], wcs),
                    output_projection=header,
                    shape_out=final_output_shape_hw,
                )
            except Exception:
                fp = np.zeros(final_output_shape_hw, dtype=float)
            footprints.append(fp)
        footprints = np.stack(footprints, axis=0)

        metrics = []
        for i, arr in enumerate(data_all):
            img_ch = arr[..., 1] if arr.shape[-1] > 1 else arr[..., 0]
            try:
                star_mask = detect_stars(img_ch)
            except Exception as exc:  # pragma: no cover - photutils missing
                logger.debug("Star detection failed: %s", exc)
                star_mask = np.zeros_like(img_ch, dtype=bool)
            med, noise, grad = _background_metrics(img_ch, star_mask)
            cover = coverage_fraction(i, footprints)
            metrics.append((med, noise, grad, cover))

        global_median = float(np.median([m[0] for m in metrics])) if metrics else 0.0
        scores = []
        for i, (med, noise, grad, cover) in enumerate(metrics):
            score = abs(med - global_median) + noise + 2.0 * grad - 0.5 * cover
            scores.append(score)
        if scores:
            best_ref = int(np.argmin(scores))

        if force_reference_index is not None and 0 <= force_reference_index < len(data_all):
            best_ref = int(force_reference_index)

        for i, (met, sc) in enumerate(zip(metrics, scores)):
            flag = " <-- REF" if best_ref is not None and i == best_ref else ""
            logger.debug(
                "BG_METRICS idx=%d med=%.3f noise=%.3f grad=%.3f cover=%.3f score=%.3f%s",
                i,
                met[0],
                met[1],
                met[2],
                met[3],
                sc,
                flag,
            )
        if best_ref is not None:
            logger.debug("Selected background reference index: %d", best_ref)

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
                kwargs_local["input_weights"] = weight_arrays
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
                if auto_bg_reference and match_bg and best_ref is not None:
                    if "background_reference" in sig.parameters:
                        kwargs_local["background_reference"] = best_ref
            except Exception:
                if use_memmap and "memmap_dir" not in kwargs_local:
                    kwargs_local["memmap_dir"] = memmap_dir
                if auto_bg_reference and match_bg and best_ref is not None:
                    kwargs_local["background_reference"] = best_ref

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
