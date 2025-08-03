import logging
import os
from typing import Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from scipy.ndimage import binary_dilation

from .reproject_utils import reproject_interp

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


    header = final_output_wcs.to_header(relax=True)

    # Determine channel count from first image
    first_path, _ = master_tile_fits_with_wcs_list[0]
    with fits.open(first_path, memmap=False) as hdul:
        first = hdul[0].data.astype(np.float32)
    if first.ndim == 3 and first.shape[0] in (1, 3) and first.shape[-1] != first.shape[0]:
        first = np.moveaxis(first, 0, -1)
    if first.ndim == 2:
        first = first[..., np.newaxis]
    n_ch = first.shape[-1]
    del first

    import tempfile, psutil, gc
    _PROC = psutil.Process(os.getpid())

    def _log_ram(prefix: str = ""):
        try:
            ram = _PROC.memory_info().rss / (1024 ** 3)
            logger.info("%sRAM utilisée : %.2f Go", prefix, ram)
        except Exception:
            pass

    # Pass 1: coverage count
    coverage_count = np.zeros(final_output_shape_hw, dtype=np.uint16)
    for path, wcs in master_tile_fits_with_wcs_list:
        try:
            with fits.open(path, memmap=False) as hdul:
                arr = hdul[0].data.astype(np.float32)
        except Exception:
            continue
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] != arr.shape[0]:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        try:
            _, fp = reproject_interp((arr[..., 0], wcs), output_projection=header, shape_out=final_output_shape_hw)
        except Exception:
            fp = np.zeros(final_output_shape_hw, dtype=np.float32)
        coverage_count += (fp > 0).astype(np.uint16)
        del arr, fp
        gc.collect()

    # Pass 2: metrics and coverage fraction
    metrics = []
    medians = []
    for path, wcs in master_tile_fits_with_wcs_list:
        try:
            with fits.open(path, memmap=False) as hdul:
                arr = hdul[0].data.astype(np.float32)
        except Exception:
            continue
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] != arr.shape[0]:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]

        img_ch = arr[..., 1] if arr.shape[-1] > 1 else arr[..., 0]
        try:
            star_mask = detect_stars(img_ch)
        except Exception as exc:
            logger.debug("Star detection failed: %s", exc)
            star_mask = np.zeros_like(img_ch, dtype=bool)
        med, noise, grad = _background_metrics(img_ch, star_mask)
        try:
            _, fp = reproject_interp((arr[..., 0], wcs), output_projection=header, shape_out=final_output_shape_hw)
        except Exception:
            fp = np.zeros(final_output_shape_hw, dtype=np.float32)
        fpb = fp > 0
        cover = float(np.count_nonzero(fpb & (coverage_count >= 2))) / float(np.count_nonzero(fpb)) if np.count_nonzero(fpb) else 0.0
        metrics.append((med, noise, grad, cover))
        medians.append([float(np.median(arr[..., ch])) for ch in range(arr.shape[-1])])
        del arr, img_ch, star_mask, fp
        gc.collect()

    scores = []
    best_ref = None
    if auto_bg_reference and match_bg:
        global_med = float(np.median([m[0] for m in metrics])) if metrics else 0.0
        for m in metrics:
            scores.append(abs(m[0] - global_med) + m[1] + 2.0 * m[2] - 0.5 * m[3])
        if scores:
            best_ref = int(np.argmin(scores))
    if force_reference_index is not None and 0 <= force_reference_index < len(metrics):
        best_ref = int(force_reference_index)
    for i, (m, s) in enumerate(zip(metrics, scores if scores else [0]*len(metrics))):
        flag = " <-- REF" if best_ref is not None and i == best_ref else ""
        logger.debug("BG_METRICS idx=%d med=%.3f noise=%.3f grad=%.3f cover=%.3f score=%.3f%s", i, m[0], m[1], m[2], m[3], s, flag)
    if best_ref is not None:
        logger.debug("Selected background reference index: %d", best_ref)

    offsets = [[0.0] * n_ch for _ in metrics]
    if match_bg and best_ref is not None and medians:
        ref_med = medians[best_ref]
        for i in range(len(metrics)):
            offsets[i] = [ref_med[ch] - medians[i][ch] for ch in range(n_ch)]
            logger.debug("BG_OFFSET idx=%d %s", i, offsets[i])

    # Create memmaps for accumulation
    sum_path = tempfile.mktemp(prefix="final_sum_", suffix=".dat", dir=memmap_dir)
    wht_path = tempfile.mktemp(prefix="final_wht_", suffix=".dat", dir=memmap_dir)
    sum_mmap = np.memmap(sum_path, dtype=np.float32, mode="w+", shape=(h, w, n_ch))
    wht_mmap = np.memmap(wht_path, dtype=np.float32, mode="w+", shape=(h, w, n_ch))
    logger.info("Création du memmap %s, shape=%s", os.path.basename(sum_path), sum_mmap.shape)
    logger.info("Création du memmap %s, shape=%s", os.path.basename(wht_path), wht_mmap.shape)

    # Pass 3: reproject and accumulate
    for idx, (path, wcs) in enumerate(master_tile_fits_with_wcs_list):
        try:
            with fits.open(path, memmap=False) as hdul:
                arr = hdul[0].data.astype(np.float32)
        except Exception:
            continue
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] != arr.shape[0]:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        for ch in range(n_ch):
            img_ch = arr[..., ch] + offsets[idx][ch]
            try:
                proj, fp = reproject_interp((img_ch, wcs), output_projection=header, shape_out=final_output_shape_hw)
            except Exception:
                proj = np.zeros(final_output_shape_hw, dtype=np.float32)
                fp = np.zeros(final_output_shape_hw, dtype=np.float32)
            weight = fp
            if weight_arrays is not None:
                try:
                    w_arr = weight_arrays[idx]
                    if w_arr.ndim == 3 and w_arr.shape[0] in (1, 3) and w_arr.shape[-1] != w_arr.shape[0]:
                        w_arr = np.moveaxis(w_arr, 0, -1)
                    if w_arr.ndim == 3:
                        w_arr = w_arr[..., min(ch, w_arr.shape[-1] - 1)]
                    w_proj, w_fp = reproject_interp((w_arr, wcs), output_projection=header, shape_out=final_output_shape_hw)
                    weight = w_proj * w_fp
                except Exception:
                    weight = np.zeros(final_output_shape_hw, dtype=np.float32)
            sum_mmap[..., ch] += proj * weight
            wht_mmap[..., ch] += weight
        logger.info("Image %d/%d reprojetée et accumulée.", idx + 1, len(master_tile_fits_with_wcs_list))
        _log_ram(f"Après image {idx + 1} : ")
        del arr, proj, fp, weight
        gc.collect()

    with np.errstate(divide="ignore", invalid="ignore"):
        final = sum_mmap / np.clip(wht_mmap, 1e-8, None)
    coverage = wht_mmap[..., 0].copy()

    del sum_mmap, wht_mmap
    gc.collect()
    try:
        os.remove(sum_path)
        os.remove(wht_path)
    except Exception:
        pass
    logger.info("Suppression memmaps temporaires.")

    return final.astype(np.float32), coverage.astype(np.float32)
