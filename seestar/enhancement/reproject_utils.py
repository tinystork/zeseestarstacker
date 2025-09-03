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
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.stats import sigma_clip
from typing import Tuple, Optional
import logging
import os
import gc
import tempfile

logger = logging.getLogger(__name__)
import numpy as np

try:  # optional core helper
    from seestar.core.reprojection_utils import collect_headers as _collect_headers
except Exception:  # pragma: no cover
    _collect_headers = None  # type: ignore


def sanitize_header_for_wcs(hdr):
    for k, v in list(hdr.items()):
        if k == "CONTINUE":
            hdr[k] = str(v)
    while "HISTORY" in hdr:
        del hdr["HISTORY"]
    while "COMMENT" in hdr:
        del hdr["COMMENT"]
    return hdr


def ensure_wcs_pixel_shape(wcs_obj, height, width):
    try:
        wcs_obj.pixel_shape = (int(width), int(height))
    except Exception:
        pass
    return wcs_obj


def is_valid_celestial_wcs(wcs: WCS) -> bool:
    try:
        if not wcs.has_celestial:
            return False
        scales = proj_plane_pixel_scales(wcs.celestial)
        if not np.isfinite(scales).all():
            return False
        arcsec = scales * 3600.0
        if np.any(arcsec < 0.05) or np.any(arcsec > 20000.0):
            return False
        _ = wcs.to_header(relax=True)
        return True
    except Exception:
        return False


try:  # pragma: no cover - allow missing reproject during import
    from reproject.mosaicking import find_optimal_celestial_wcs
except Exception:  # pragma: no cover
    find_optimal_celestial_wcs = None


def compute_final_output_grid_from_wcs(wcs_list, shape_list, auto_rotate=True):
    if find_optimal_celestial_wcs is None:
        raise ImportError(
            "The 'reproject' package is required for this functionality. "
            "Please install it with 'pip install reproject'."
        )

    valid = [
        (w, s)
        for (w, s) in zip(wcs_list, shape_list)
        if is_valid_celestial_wcs(w)
    ]
    if not valid:
        raise RuntimeError("No valid WCS available to build the global grid.")
    w_ok, s_ok = zip(*valid)
    inputs = [(s, w) for (w, s) in zip(w_ok, s_ok)]
    out_wcs, shape_out = find_optimal_celestial_wcs(
        inputs, auto_rotate=auto_rotate
    )
    return out_wcs, shape_out, len(wcs_list) - len(valid)

def compute_final_output_grid(headers, auto_rotate=True):
    wcs_list = []
    shapes = []
    for hdr in headers:
        try:
            hdr = sanitize_header_for_wcs(hdr)
            w = WCS(hdr, naxis=2)
            h = int(hdr.get("NAXIS2"))
            w_pix = int(hdr.get("NAXIS1"))
            wcs_list.append(w)
            shapes.append((h, w_pix))
        except Exception:
            continue
    out_wcs, shape_out, _ = compute_final_output_grid_from_wcs(
        wcs_list, shapes, auto_rotate=auto_rotate
    )
    return out_wcs, shape_out


def subtract_sigma_clipped_median(img, min_valid: int = 1024):
    mask = np.isfinite(img)
    if int(np.count_nonzero(mask)) < int(min_valid):
        return img, 0.0
    clipped = sigma_clip(img[mask], sigma=3.0, maxiters=5)
    med = np.nanmedian(clipped.filled(np.nan))
    med_val = 0.0 if not np.isfinite(med) else float(med)
    return img - med_val, med_val


def _subtract_sky_median(image, nsig=3.0, maxiters=5, min_valid: int = 1024):
    mask = np.isfinite(image)
    if int(np.count_nonzero(mask)) < int(min_valid):
        return image
    clipped = sigma_clip(image[mask], sigma=nsig, maxiters=maxiters)
    med = np.nanmedian(clipped.filled(np.nan))
    med_val = 0.0 if not np.isfinite(med) else float(med)
    return image - med_val



class ReprojectCoaddResult:
    def __init__(self, image, weight, wcs):
        self.image = image
        self.weight = weight
        self.wcs = wcs

    def __iter__(self):
        yield self.image
        yield self.weight


def _estimate_mem_gb(shape_out, n_maps=2):
    h, w = int(shape_out[0]), int(shape_out[1])
    bytes_total = h * w * 4 * n_maps
    return bytes_total / (1024**3)


# [B1-COADD-FIX] Garantir une image 2D pour la reprojection
def _ensure_2d(img: np.ndarray) -> np.ndarray:  # [B1-COADD-FIX]
    """
    Force une image 2D. Si HWC (RGB), lever si cette fonction est appelée
    au mauvais endroit (la séparation par canal doit être faite en amont),
    ou retourner le premier canal si la logique l'exige.
    """  # [B1-COADD-FIX]
    if img is None:  # [B1-COADD-FIX]
        raise ValueError("[B1-COADD-FIX] img is None")  # [B1-COADD-FIX]
    arr = np.asarray(img)  # [B1-COADD-FIX]
    if arr.ndim == 2:  # [B1-COADD-FIX]
        return arr  # [B1-COADD-FIX]
    if arr.ndim == 3:  # [B1-COADD-FIX]
        chw_like = (arr.shape[0] in (1, 3, 4)) and (arr.shape[-1] not in (1, 3, 4))
        if chw_like:
            logger.warning(
                "[B1-COADD-FIX] 3D CHW array; taking channel 0. shape=%s", arr.shape
            )
            return arr[0, ...]
        else:
            logger.warning(
                "[B1-COADD-FIX] 3D HWC array; taking channel 0. shape=%s", arr.shape
            )
            return arr[..., 0]
    raise ValueError(
        f"[B1-COADD-FIX] Unsupported ndim={arr.ndim} for reprojection input."
    )  # [B1-COADD-FIX]


def _open_fits_safely(path):
    """Open a FITS file avoiding memmap scaling issues.

    The file is first inspected with ``memmap=True``. If scaling keywords
    (``BZERO``, ``BSCALE`` or ``BLANK``) are present the file is reopened with
    ``memmap=False`` and ``do_not_scale_image_data=False`` so that astropy can
    safely apply the required scaling. Otherwise a memory-mapped array is
    returned.
    """

    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        hdr = hdul[0].header.copy()
        needs_scaling = any(k in hdr for k in ("BZERO", "BSCALE", "BLANK"))
        if not needs_scaling:
            return hdul[0].data, hdr

    with fits.open(path, memmap=False, do_not_scale_image_data=False) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header.copy()
    return data, hdr


def _to_chw(data: np.ndarray) -> np.ndarray:
    """Normalize ``data`` to ``(C, H, W)`` form.

    Handles grayscale ``(H, W)``, ``(C, H, W)`` and ``(H, W, C)`` inputs.
    Unknown layouts fall back to a single-channel view.
    """

    if data.ndim == 2:
        return data[None, ...]
    if data.shape[0] in (3, 4):
        return data
    if data.shape[-1] in (3, 4):
        return np.transpose(data, (2, 0, 1))
    return data[None, ...]


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

    kwargs = dict(kwargs)  # [B1-COADD-FIX]
    # Ne jamais transmettre ces flags UI/streaming au moteur de reprojection
    kwargs.pop("return_footprint", None)      # ok pour astropy
    kwargs.pop("crop_to_footprint", None)     # sinon -> TypeError dans reproject_interp

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

    use_astropy = _astropy_reproject_and_coadd is not None

    if use_astropy:
        mem_threshold = float(os.environ.get("REPROJECT_MEM_THRESHOLD_GB", "8"))
        mem_required = np.prod(shape_out) * 2 * 8 / 1024**3
        if mem_required > mem_threshold:
            logger.info(
                "Disabling astropy reproject_and_coadd (%.1f GiB required > %.1f GiB)",
                mem_required,
                mem_threshold,
            )
            use_astropy = False

    if use_astropy:
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

    sum_image = np.zeros(shape_out, dtype=np.float64)  # [B1-COADD-FIX]
    cov_image = np.zeros(shape_out, dtype=np.float64)  # [B1-COADD-FIX]

    kept = 0  # [B1-COADD-FIX]
    total = 0  # [B1-COADD-FIX]

    for (img, wcs_in), weight in zip(filtered_pairs, filtered_weights):
        total += 1  # [B1-COADD-FIX]
        img2d = _ensure_2d(np.asarray(img))  # [B1-COADD-FIX]

        proj_img, footprint = reproject_function(
            (img2d, wcs_in),
            output_projection=ref_wcs,
            shape_out=shape_out,
            return_footprint=True,
            **kwargs,
        )  # [B1-COADD-FIX]

        proj_img = np.nan_to_num(
            proj_img, nan=0.0, posinf=0.0, neginf=0.0, copy=False
        )  # [B1-COADD-FIX]
        footprint = np.nan_to_num(
            footprint, nan=0.0, posinf=0.0, neginf=0.0, copy=False
        )  # [B1-COADD-FIX]

        if weight is None:  # [B1-COADD-FIX]
            weight_proj = footprint  # [B1-COADD-FIX]
        elif np.isscalar(weight):  # [B1-COADD-FIX]
            weight_proj = footprint * float(weight)  # [B1-COADD-FIX]
        else:
            weight = np.asarray(weight)  # [B1-COADD-FIX]
            if weight.shape != img2d.shape:  # [B1-COADD-FIX]
                raise ValueError("[B1-COADD-FIX] weight shape mismatch")  # [B1-COADD-FIX]
            w_reproj, w_fp = reproject_function(
                (weight, wcs_in),
                output_projection=ref_wcs,
                shape_out=shape_out,
                return_footprint=True,
                **kwargs,
            )  # [B1-COADD-FIX]
            weight_proj = w_reproj * w_fp  # [B1-COADD-FIX]

        weight_proj = np.nan_to_num(
            weight_proj, nan=0.0, posinf=0.0, neginf=0.0, copy=False
        )  # [B1-COADD-FIX]

        if np.any(footprint > 0):  # [B1-COADD-FIX]
            proj_img = np.nan_to_num(
                proj_img, nan=0.0, posinf=0.0, neginf=0.0, copy=False
            )
            weight_proj = np.nan_to_num(
                weight_proj, nan=0.0, posinf=0.0, neginf=0.0, copy=False
            )
            sum_image += proj_img * weight_proj  # [B1-COADD-FIX]
            cov_image += weight_proj  # [B1-COADD-FIX]
            kept += 1  # [B1-COADD-FIX]
        else:
            logger.debug("[B1-COADD-FIX] Skipped entry (zero footprint).")  # [B1-COADD-FIX]

        del proj_img, footprint, weight_proj  # [B1-COADD-FIX]

    out = np.divide(
        sum_image,
        cov_image,
        out=np.zeros_like(sum_image, dtype=np.float32),
        where=(cov_image > 0),
    )  # [B1-COADD-FIX]
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  # [B1-COADD-FIX]

    try:
        logger.info(
            "[B1-COADD-FIX] coadd stats: kept=%d/%d, cov>0=%d, cov_sum=%.3f, sum_min=%.3g, sum_max=%.3g",
            kept,
            total,
            int(np.count_nonzero(valid)),
            float(cov_image.sum()),
            float(np.nanmin(sum_image)),
            float(np.nanmax(sum_image)),
        )
    except Exception:  # [B1-COADD-FIX]
        pass  # [B1-COADD-FIX]

    return out, cov_image.astype(np.float32)  # [B1-COADD-FIX]


def reproject_and_coadd_from_paths(
    paths,
    output_projection=None,
    shape_out=None,
    match_background=True,
    tile_size=None,
    prefer_streaming_fallback=False,
    subtract_sky_median=True,
    **kwargs,
):
    """Load FITS files from ``paths`` then call :func:`reproject_and_coadd`.

    Parameters
    ----------
    paths : iterable of str
        FITS file paths containing valid WCS information.
    output_projection : astropy.wcs.WCS or FITS header, optional
        Target projection. When ``None`` the WCS of all files are combined.
    shape_out : tuple, optional
        Output shape ``(H, W)``. When ``None`` it is derived from the input
        headers.
    match_background : bool, optional
        Deprecated alias for ``subtract_sky_median``.
    prefer_streaming_fallback : bool, optional
        If ``True`` and the required memory exceeds the threshold, fall back to
        :func:`streaming_reproject_and_coadd`.
    subtract_sky_median : bool, optional
        If ``True`` subtract a sigma-clipped median from each image prior to
        reprojection.
    tile_size : int, optional
        Forwarded to :func:`streaming_reproject_and_coadd` when used.
    **kwargs : dict
        Forwarded to :func:`reproject_and_coadd`.
    """

    kwargs = dict(kwargs)
    kwargs.pop("return_footprint", None)
    crop_to_footprint = kwargs.pop("crop_to_footprint", False)

    if match_background is not None:
        subtract_sky_median = match_background

    if shape_out is not None:
        mem_threshold = float(os.environ.get("REPROJECT_MEM_THRESHOLD_GB", "8"))
        mem_gb = _estimate_mem_gb(shape_out, n_maps=2)
        if mem_gb > mem_threshold and not prefer_streaming_fallback:
            raise MemoryError(
                f"Requested output grid {shape_out} requires {mem_gb:.1f} GiB (> {mem_threshold:.1f} GiB)."
            )

    if output_projection is None or shape_out is None:
        if _collect_headers is not None:
            infos = _collect_headers(paths)
            shape_list = [sh for sh, _ in infos]
            wcs_list = [w for _, w in infos]
        else:  # pragma: no cover - fallback when core utils missing
            wcs_list = []
            shape_list = []
            for fp in paths:
                try:
                    hdr = fits.getheader(fp, memmap=False)
                    w = WCS(hdr, naxis=2)
                    shape_list.append((int(hdr.get("NAXIS2")), int(hdr.get("NAXIS1"))))
                    wcs_list.append(w)
                except Exception:
                    continue

        try:
            out_wcs, shape_out, dropped = compute_final_output_grid_from_wcs(
                wcs_list, shape_list, auto_rotate=True
            )
        except RuntimeError:
            logger.warning(
                "[Reproject] No valid WCS – falling back to streaming with internal grid"
            )
            return streaming_reproject_and_coadd(
                paths,
                output_wcs=None,
                shape_out=None,
                subtract_sky_median=subtract_sky_median,
                tile_size=tile_size or 1024,
                crop_to_footprint=crop_to_footprint,
            )

    else:
        out_wcs = output_projection if isinstance(output_projection, WCS) else WCS(output_projection)
        dropped = 0

    shape_out = tuple(int(round(x)) for x in shape_out)
    if shape_out[0] <= 0 or shape_out[1] <= 0:
        raise ValueError(f"invalid shape_out: {shape_out}")

    mem_threshold = float(os.environ.get("REPROJECT_MEM_THRESHOLD_GB", "8"))
    mem_gb = _estimate_mem_gb(shape_out, n_maps=2)
    logger.info(
        f"[Reproject] Global grid {shape_out}, est. mem ~{mem_gb:.2f} GiB; dropped_wcs={dropped}"
    )

    if prefer_streaming_fallback and mem_gb > mem_threshold:
        logger.warning(
            f"[Reproject] Grid exceeds {mem_threshold} GiB → fallback to streaming."
        )
        return streaming_reproject_and_coadd(
            paths,
            output_wcs=out_wcs,
            shape_out=shape_out,
            subtract_sky_median=subtract_sky_median,
            tile_size=tile_size or 1024,
            crop_to_footprint=crop_to_footprint,
        )

    pairs = []
    for fp in paths:
        try:
            with fits.open(fp, memmap=False) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float32)
                hdr = hdul[0].header
        except Exception:
            logger.warning("Skipping invalid FITS '%s' for reprojection", fp, exc_info=True)
            continue
        sanitize_header_for_wcs(hdr)
        wcs = WCS(hdr, naxis=2)
        h, w = data.shape[:2]
        ensure_wcs_pixel_shape(wcs, h, w)
        if data.ndim == 3 and data.shape[0] in (1, 3) and data.shape[-1] != data.shape[0]:
            data = np.moveaxis(data, 0, -1)
        if subtract_sky_median:
            if data.ndim == 2:
                data, _ = subtract_sigma_clipped_median(data, min_valid=1024)
            elif data.ndim == 3:
                for c in range(data.shape[-1]):
                    data[..., c], _ = subtract_sigma_clipped_median(
                        data[..., c], min_valid=1024
                    )
        pairs.append((data, wcs))
        logger.debug("[B1-COADD-FIX] input=%s ndim=%d shape=%s", fp, data.ndim, data.shape)

    if not pairs:
        raise RuntimeError("Reproject requested but no aligned FITS were produced.")

    first = pairs[0][0]
    if first.ndim == 3:
        channels = []
        cov_out = None
        C = first.shape[-1]
        for c in range(C):
            ch_pairs = [(img[:, :, c], wcs) for img, wcs in pairs]
            sci, cov = reproject_and_coadd(
                ch_pairs, out_wcs, shape_out, **kwargs
            )
            channels.append(sci)
            if cov_out is None:
                cov_out = cov
        result = np.stack(channels, axis=-1)
        return ReprojectCoaddResult(result, cov_out, out_wcs)

    sci, cov = reproject_and_coadd(pairs, out_wcs, shape_out, **kwargs)
    return ReprojectCoaddResult(sci, cov, out_wcs)


def streaming_reproject_and_coadd(
    paths,
    reference_path=None,
    output_path=None,
    tile_size=1024,
    dtype_out=np.float32,
    memmap_dir=None,
    keep_intermediates=False,
    subtract_sky_median=True,
    reproject_function=None,
    output_wcs=None,
    shape_out=None,
    crop_to_footprint=True,
    match_background=None,
):
    """Streamingly reproject ``paths`` and coadd them on disk.

    Parameters
    ----------
    paths : list of str
        Paths to aligned FITS files.
    reference_path : str, optional
        FITS file providing the target WCS. Defaults to first path.
    output_path : str, optional
        Destination FITS file. When ``None`` the function returns ``False``.
    tile_size : int, optional
        Height/width of processing tiles.
    dtype_out : numpy dtype, optional
        Output dtype for the final stack.
    memmap_dir : str, optional
        Directory where temporary memmaps are stored.
    keep_intermediates : bool, optional
        If ``True`` temporary memmaps are preserved.
    match_background : bool, optional
        If ``True`` a robust median background is subtracted from each input
        before reprojection.
    reproject_function : callable, optional
        Reprojection function, defaults to :func:`reproject.reproject_interp`.
    output_wcs : :class:`astropy.wcs.WCS`, optional
        Precomputed output WCS describing the mosaic grid.
    shape_out : tuple of int, optional
        Explicit output ``(H, W)`` shape corresponding to ``output_wcs``.
    crop_to_footprint : bool, optional
        If ``True`` crop the final mosaic to the region where the weight map is
        non-zero before writing to ``output_path``.
    """

    if match_background is not None:
        subtract_sky_median = match_background

    if reproject_function is None:
        reproject_function = _reproject_interp

    if not paths:
        return ReprojectCoaddResult(np.array([]), np.array([]), None)

    paths_for_channels = list(paths)
    if reference_path and reference_path not in paths_for_channels:
        paths_for_channels.append(reference_path)
    C_out = 1
    for fp in paths_for_channels:
        try:
            data_tmp, _ = _open_fits_safely(fp)
        except Exception:
            continue
        C_out = max(C_out, _to_chw(data_tmp).shape[0])

    ref_fp = reference_path or paths[0]
    if output_wcs is not None:
        out_wcs = output_wcs.deepcopy() if hasattr(output_wcs, "deepcopy") else output_wcs
        try:
            _, ref_hdr = _open_fits_safely(ref_fp)
        except Exception:
            ref_hdr = fits.Header()
    else:
        try:
            ref_data_raw, ref_hdr = _open_fits_safely(ref_fp)
            ref_chw = _to_chw(ref_data_raw)
            if shape_out is None:
                shape_out = ref_chw.shape[1:]  # (H, W)
        except Exception:
            ref_hdr = fits.Header()
            if shape_out is None:
                shape_out = None
        sanitize_header_for_wcs(ref_hdr)
        out_wcs = WCS(ref_hdr, naxis=2)

    if shape_out is None:
        if out_wcs.pixel_shape is not None and out_wcs.array_shape is not None:
            shape_out = tuple(out_wcs.array_shape)
        else:
            img_tmp, _ = _open_fits_safely(ref_fp)
            shape_out = _to_chw(img_tmp).shape[1:]

    if out_wcs.pixel_shape is None or out_wcs.array_shape is None:
        out_wcs.pixel_shape = (shape_out[1], shape_out[0])
        out_wcs.array_shape = shape_out

    try:  # [B1-COADD-FIX]
        logger.info(
            "[B1-COADD-FIX] streaming grid shape=%s C_out=%d", shape_out, C_out
        )
    except Exception:  # pragma: no cover - logging should not fail
        pass

    bg_medians = {}
    if subtract_sky_median:
        for fp in paths:
            try:
                data_bg, _ = _open_fits_safely(fp)
                chw_bg = _to_chw(data_bg)
                meds = []
                for c in range(chw_bg.shape[0]):
                    arr_c = chw_bg[c]
                    mask_c = np.isfinite(arr_c)
                    if np.count_nonzero(mask_c) >= 1024:
                        clipped = sigma_clip(arr_c[mask_c], sigma=3.0, maxiters=5)
                        med = np.nanmedian(clipped.filled(np.nan))
                        med_val = 0.0 if not np.isfinite(med) else float(med)
                    else:
                        med_val = 0.0
                    meds.append(med_val)
                bg_medians[fp] = np.asarray(meds)[:, None, None]
            except Exception:
                logger.warning(
                    "Background estimation failed for '%s'", fp, exc_info=False
                )

    if memmap_dir is None:
        memmap_dir = tempfile.mkdtemp(prefix="stream_reproject_")
    os.makedirs(memmap_dir, exist_ok=True)
    sum_path = os.path.join(memmap_dir, "sum_map.memmap")
    wht_path = os.path.join(memmap_dir, "wht_map.memmap")
    if C_out == 1:
        sum_map = np.memmap(sum_path, dtype=np.float64, mode="w+", shape=shape_out)
        wht_map = np.memmap(wht_path, dtype=np.float32, mode="w+", shape=shape_out)
    else:
        sum_map = np.memmap(sum_path, dtype=np.float64, mode="w+", shape=(C_out, *shape_out))
        wht_map = np.memmap(wht_path, dtype=np.float32, mode="w+", shape=(C_out, *shape_out))
    sum_map[:] = 0.0
    wht_map[:] = 0.0

    tile_h = tile_w = int(tile_size)

    for y0 in range(0, shape_out[0], tile_h):
        y1 = min(y0 + tile_h, shape_out[0])
        for x0 in range(0, shape_out[1], tile_w):
            x1 = min(x0 + tile_w, shape_out[1])
            sub_wcs = out_wcs.slice((slice(y0, y1), slice(x0, x1)))
            for fp in paths:
                try:
                    data, hdr = _open_fits_safely(fp)
                    # Keep raw dimensionality; _to_chw will normalize to (C, H, W)
                except Exception:
                    logger.warning("Skipping invalid FITS '%s' for streaming reprojection", fp)
                    continue
                sanitize_header_for_wcs(hdr)
                in_wcs = WCS(hdr, naxis=2)
                chw_in = _to_chw(data)
                if subtract_sky_median and fp in bg_medians:
                    bg = bg_medians[fp]
                    if bg.shape[0] != chw_in.shape[0]:
                        if bg.shape[0] == 1 and chw_in.shape[0] > 1:
                            bg = np.repeat(bg, chw_in.shape[0], axis=0)
                        elif bg.shape[0] > 1 and chw_in.shape[0] == 1:
                            bg = np.mean(bg, axis=0, keepdims=True)
                        else:
                            bg = bg[: chw_in.shape[0]]
                    chw_in = chw_in - bg
                if chw_in.shape[0] != C_out:
                    if chw_in.shape[0] == 1 and C_out > 1:
                        # broadcast grayscale -> RGB-like
                        chw_in = np.repeat(chw_in, C_out, axis=0)
                    elif chw_in.shape[0] > 1 and C_out == 1:
                        # collapse RGB -> mono by averaging
                        chw_in = np.mean(chw_in, axis=0, keepdims=True)
                    else:
                        logger.warning("Channel mismatch for '%s'", fp)
                        continue
                for c in range(C_out):
                    try:
                        arr, footprint = reproject_function(
                            (chw_in[c], in_wcs),
                            output_projection=sub_wcs,
                            shape_out=(y1 - y0, x1 - x0),
                            return_footprint=True,
                        )
                    except Exception:
                        logger.warning("Reprojection failed for '%s' (channel %d)", fp, c, exc_info=True)
                        continue
                    arr = arr.astype(dtype_out, copy=False)
                    footprint = footprint.astype(np.float32, copy=False)

                    np.nan_to_num(
                        arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    np.nan_to_num(
                        footprint, copy=False, nan=0.0, posinf=0.0, neginf=0.0
                    )

                    arr *= footprint
                    np.nan_to_num(
                        arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0
                    )

                    try:
                        logger.debug(
                            "[B1-COADD-FIX] tile=(%d:%d,%d:%d) finite(arr)=%s finite(footprint)=%s min/max(arr)=(%.3g,%.3g)",
                            y0,
                            y1,
                            x0,
                            x1,
                            bool(np.isfinite(arr).all()),
                            bool(np.isfinite(footprint).all()),
                            float(np.nanmin(arr)) if arr.size else float('nan'),
                            float(np.nanmax(arr)) if arr.size else float('nan'),
                        )
                    except Exception:  # pragma: no cover
                        pass

                    if C_out == 1:
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                        footprint = np.nan_to_num(
                            footprint, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        sum_map[y0:y1, x0:x1] += arr
                        wht_map[y0:y1, x0:x1] += footprint
                    else:
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                        footprint = np.nan_to_num(
                            footprint, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        sum_map[c, y0:y1, x0:x1] += arr
                        wht_map[c, y0:y1, x0:x1] += footprint
                    del arr, footprint
                gc.collect()

    final_path = os.path.join(memmap_dir, "final.memmap")
    if C_out == 1:
        final_map = np.memmap(final_path, dtype=dtype_out, mode="w+", shape=shape_out)
    else:
        final_map = np.memmap(final_path, dtype=dtype_out, mode="w+", shape=(C_out, *shape_out))
    final_raw = final_map
    if C_out == 1:
        for y0 in range(0, shape_out[0], tile_h):
            y1 = min(y0 + tile_h, shape_out[0])
            for x0 in range(0, shape_out[1], tile_w):
                s = sum_map[y0:y1, x0:x1]
                w = wht_map[y0:y1, x0:x1]
                tile_out = final_map[y0:y1, x0:x1]
                tile_out.fill(0.0)
                np.divide(s, w, out=tile_out, where=(w > 0))
                np.nan_to_num(
                    tile_out, copy=False, nan=0.0, posinf=0.0, neginf=0.0
                )
    else:
        for c in range(C_out):
            for y0 in range(0, shape_out[0], tile_h):
                y1 = min(y0 + tile_h, shape_out[0])
                for x0 in range(0, shape_out[1], tile_w):
                    s = sum_map[c, y0:y1, x0:x1]
                    w = wht_map[c, y0:y1, x0:x1]
                    tile_out = final_map[c, y0:y1, x0:x1]
                    tile_out.fill(0.0)
                    np.divide(s, w, out=tile_out, where=(w > 0))
                    np.nan_to_num(
                        tile_out, copy=False, nan=0.0, posinf=0.0, neginf=0.0
                    )

    np.nan_to_num(final_map, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    try:  # [B1-COADD-FIX] log statistics for debugging
        sum_min, sum_max = float(np.nanmin(sum_map)), float(np.nanmax(sum_map))
        wht_min, wht_max = float(np.nanmin(wht_map)), float(np.nanmax(wht_map))
        fin_min, fin_max = float(np.nanmin(final_map)), float(np.nanmax(final_map))
        logger.info(
            "[B1-COADD-FIX] stats: shape=%s C_out=%d sum_minmax=(%.3g,%.3g) wht_minmax=(%.3g,%.3g) final_minmax=(%.3g,%.3g)",
            shape_out,
            C_out,
            sum_min,
            sum_max,
            wht_min,
            wht_max,
            fin_min,
            fin_max,
        )
    except Exception:  # pragma: no cover - stats are best effort
        wht_max = float(np.nanmax(wht_map))

    if float(wht_max) == 0:
        sum_map.flush(); wht_map.flush(); final_raw.flush()
        if not keep_intermediates:
            try:
                os.remove(sum_path)
                os.remove(wht_path)
                os.remove(final_path)
                os.rmdir(memmap_dir)
            except Exception:
                pass
        raise RuntimeError("Aucune contribution reçue (mismatch de canaux)")

    if crop_to_footprint:
        if C_out == 1:
            mask = wht_map > 0
        else:
            mask = np.sum(wht_map, axis=0) > 0
        ys, xs = np.where(mask)
        if ys.size > 0 and xs.size > 0:
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            if C_out == 1:
                final_map = final_map[y0:y1, x0:x1]
            else:
                final_map = final_map[:, y0:y1, x0:x1]
            out_wcs.wcs.crpix -= [x0, y0]
            out_wcs.array_shape = (y1 - y0, x1 - x0)
            out_wcs.pixel_shape = (x1 - x0, y1 - y0)
            try:
                out_wcs._naxis1 = x1 - x0
                out_wcs._naxis2 = y1 - y0
            except Exception:
                pass
            shape_out = (y1 - y0, x1 - x0)

    try:
        if C_out == 1:
            logger.info(
                "Channel stats: sum[min=%.3g,max=%.3g] wht[min=%.3g,max=%.3g] final[min=%.3g,max=%.3g]",
                float(np.nanmin(sum_map)),
                float(np.nanmax(sum_map)),
                float(np.nanmin(wht_map)),
                float(np.nanmax(wht_map)),
                float(np.nanmin(final_map)),
                float(np.nanmax(final_map)),
            )
        else:
            for c in range(C_out):
                logger.info(
                    "Channel %d stats: sum[min=%.3g,max=%.3g] wht[min=%.3g,max=%.3g] final[min=%.3g,max=%.3g]",
                    c,
                    float(np.nanmin(sum_map[c])),
                    float(np.nanmax(sum_map[c])),
                    float(np.nanmin(wht_map[c])),
                    float(np.nanmax(wht_map[c])),
                    float(np.nanmin(final_map[c])),
                    float(np.nanmax(final_map[c])),
                )
    except Exception:
        pass

    if output_path is not None:
        hdr_out = out_wcs.to_header(relax=True)
        data_out = final_map
        if data_out.ndim == 3 and data_out.shape[-1] in (3, 4) and data_out.shape[0] not in (1, 3, 4):
            data_out = np.moveaxis(data_out, -1, 0)
        hdu = fits.PrimaryHDU(data=data_out.astype(np.float32), header=hdr_out)
        fits.HDUList([hdu]).writeto(output_path, overwrite=True)
        logger.info(
            "Streaming coadd written: shape=%s dtype=%s",
            getattr(data_out, "shape", None),
            data_out.dtype,
        )

    sum_map.flush(); wht_map.flush(); final_raw.flush()
    result_arr = np.asarray(final_map)
    wht_arr = np.asarray(wht_map)
    if not keep_intermediates:
        try:
            os.remove(sum_path)
            os.remove(wht_path)
            os.remove(final_path)
            os.rmdir(memmap_dir)
        except Exception:
            pass
    return ReprojectCoaddResult(result_arr, wht_arr, out_wcs)


reproject_interp = _reproject_interp

__all__ = [
    "reproject_and_coadd",
    "reproject_interp",
    "reproject_and_coadd_from_paths",
    "streaming_reproject_and_coadd",
]
