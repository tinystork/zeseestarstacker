import argparse
import csv
import os
import sys
import shutil

# Ensure console encoding supports UTF-8 characters (e.g. emojis)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# When executed directly, ensure the package root is discoverable.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import gc
import ctypes
import logging
import time
from typing import List, Dict, Tuple

import psutil

import numpy as np
from astropy.io import fits
import cv2
import imageio
from numpy.lib.format import open_memmap
from astropy.wcs import WCS
from seestar.core.alignment import SeestarAligner

logger = logging.getLogger(__name__)
aligner = None


def _init_logger(out_dir: str) -> None:
    log_path = os.path.join(out_dir, "boring_stack.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _format_seconds(secs: float) -> str:
    """Return HH:MM:SS string for ``secs`` seconds."""
    secs = max(0, int(secs))
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _safe_print(*args, **kwargs) -> None:
    """Print without raising if the output stream is closed."""
    try:
        print(*args, **kwargs)
    except (OSError, ValueError) as e:
        logger.debug("stdout/stderr unavailable: %s", e)


logger = logging.getLogger(__name__)


try:
    if "--out" in sys.argv:
        _init_logger(sys.argv[sys.argv.index("--out") + 1])
except Exception:
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(message)s")

# Allow running as a standalone script
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from seestar.core.stack_methods import (
        _stack_kappa_sigma,
        _stack_linear_fit_clip,
        _stack_mean,
        _stack_median,
        _stack_winsorized_sigma,
    )
else:
    from seestar.core.stack_methods import (
        _stack_kappa_sigma,
        _stack_linear_fit_clip,
        _stack_mean,
        _stack_median,
        _stack_winsorized_sigma,
    )


def _reproj():
    from seestar import reproject_utils as ru

    return ru


def solve_local_plate(path: str):
    """Return WCS from FITS header if present."""
    try:
        hdr = fits.getheader(path, memmap=False)
        wcs = WCS(hdr)
        return wcs if wcs.is_celestial else None
    except Exception:
        return None


def get_wcs_from_astap(path: str):
    """Load ``path``.wcs if it exists."""
    wcs_path = os.path.splitext(path)[0] + ".wcs"
    if os.path.isfile(wcs_path):
        try:
            hdr = fits.getheader(wcs_path, memmap=False)
            return WCS(hdr)
        except Exception:
            return None
    return None


def solve_with_astrometry_local(path: str):
    """Attempt local astrometry.net solve-field via zemosaic."""
    try:
        from zemosaic import zemosaic_astrometry as za

        hdr = fits.getheader(path, memmap=False)
        cfg = os.environ.get("ANSVR_CONFIG", "")
        if not cfg:
            return None
        return za.solve_with_ansvr(path, hdr, cfg)
    except Exception:
        return None


def solve_with_astrometry_net(path: str, api_key: str):
    """Call astrometry.net web solve via zemosaic."""
    from zemosaic import zemosaic_astrometry as za

    hdr = fits.getheader(path, memmap=False)
    return za.solve_with_astrometry_net(path, hdr, api_key)


def warp_image(img: np.ndarray, wcs_in: WCS, wcs_ref: WCS, shape_ref: tuple[int, int]):
    from seestar.core.reprojection import reproject_to_reference_wcs

    return reproject_to_reference_wcs(img, wcs_in, wcs_ref, shape_ref)


def to_hwc(arr: np.ndarray, hdr: fits.Header | None = None) -> np.ndarray:
    """Return ``arr`` in ``(H, W, C)`` order if necessary."""
    if arr.ndim == 2:
        # Convert grayscale planes to explicit channel dimension
        return arr[..., None]

    if arr.ndim == 3:
        # Typical case: channel-first (C, H, W)
        if arr.shape[0] <= 4:
            return arr.transpose(1, 2, 0)

        # Less common: (W, H, C) with header confirming orientation
        if (
            hdr is not None
            and arr.shape[2] <= 4
            and hdr.get("NAXIS1") == arr.shape[0]
            and hdr.get("NAXIS2") == arr.shape[1]
        ):
            return arr.transpose(1, 0, 2)

    return arr


def parse_args():
    p = argparse.ArgumentParser(description="Disk-based stacking script")
    p.add_argument("--csv", required=True, help="CSV with file list")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--tile", type=int, default=512, help="Tile height")
    p.add_argument("--kappa", type=float, default=3.0, help="Kappa value")
    p.add_argument("--winsor", type=float, default=0.05, help="Winsor limit")
    p.add_argument(
        "--max-mem",
        type=float,
        default=None,
        help="Maximum memory per stack in GB (overrides SEESTAR_MAX_MEM)",
    )
    p.add_argument("--api-key", default=None, help="Astrometry.net API key")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p.add_argument(
        "--norm",
        default="none",
        choices=["linear_fit", "sky_mean", "none"],
        help="Normalization method",
    )
    p.add_argument(
        "--weight",
        default="none",
        choices=["snr", "stars", "none", "noise_variance", "noise_fwhm", "quality"],
        help="Weighting method",
    )
    p.add_argument(
        "--reject",
        default="winsorized_sigma",
        choices=["kappa_sigma", "winsorized_sigma", "none"],
        help="Pixel rejection algorithm",
    )
    p.add_argument(
        "--no-hot-pixels",
        dest="correct_hot_pixels",
        action="store_false",
        help="Disable hot pixel correction",
    )
    p.add_argument(
        "--hot-threshold",
        type=float,
        default=3.0,
        help="Hot pixel sigma threshold",
    )
    p.add_argument(
        "--hot-neighborhood",
        type=int,
        default=5,
        help="Hot pixel neighbourhood size",
    )
    p.add_argument(
        "--use-weighting",
        action="store_true",
        default=False,
        help="Enable quality based weighting",
    )
    p.add_argument(
        "--snr-exp",
        type=float,
        default=1.0,
        help="Exponent for SNR weighting",
    )
    p.add_argument(
        "--stars-exp",
        type=float,
        default=0.5,
        help="Exponent for star weighting",
    )
    p.add_argument(
        "--min-weight",
        type=float,
        default=0.1,
        help="Minimum weight value",
    )
    p.add_argument(
        "--use-solver",
        dest="use_solver",
        action="store_true",
        help="Use third-party solver for WCS alignment",
    )
    p.add_argument(
        "--no-solver",
        dest="use_solver",
        action="store_false",
        help="Disable third-party solver and skip reprojection",
    )
    p.add_argument(
        "--cleanup-temp-files",
        dest="cleanup_temp_files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cleanup temporary aligned files after processing",
    )
    p.set_defaults(use_solver=True)
    return p.parse_args()


def read_rows(csv_path):
    """Read rows from ``csv_path`` with optional headers.

    The CSV may contain a simple list of paths or a full ``stack_plan.csv``
    with additional columns.  In the latter case the ``file_path`` column is
    used.  A header row like ``order,file`` or ``index,file`` is also
    supported.  Relative paths are resolved against the CSV location.
    """

    logger.info("lecture de stack_plan.csv: %s", csv_path)

    rows_out: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return rows_out

    header = [c.strip().lower() for c in rows[0]]
    file_idx = None
    weight_idx = None
    data_rows = rows

    if "file_path" in header:
        file_idx = header.index("file_path")
        weight_idx = header.index("weight") if "weight" in header else None
        data_rows = rows[1:]
    else:
        has_header = any(h in {"order", "file", "filename", "path", "index", "weight"} for h in header)
        if has_header:
            data_rows = rows[1:]
            if "weight" in header:
                weight_idx = header.index("weight")

    base_dir = os.path.dirname(csv_path)

    for row in data_rows:
        if not row:
            continue

        if file_idx is not None:
            if len(row) <= file_idx:
                continue
            cell = row[file_idx].strip()
        else:
            cell = row[0].strip()
            if cell.isdigit() and len(row) > 1:
                cell = row[1].strip()

        if not cell or cell.lower() in {
            "order",
            "file",
            "filename",
            "path",
            "index",
            "file_path",
        }:
            continue

        if not os.path.isabs(cell):
            cell = os.path.join(base_dir, cell)

        weight = ""
        if weight_idx is not None and len(row) > weight_idx:
            weight = row[weight_idx].strip()
        rows_out.append({"path": cell, "weight": weight})

    return rows_out


def read_paths(csv_path):
    """Return list of paths only (legacy helper)."""
    return [r["path"] for r in read_rows(csv_path)]


def get_image_shape(path):
    """Return (height, width, channels) for an image file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".fit", ".fits"}:
        with fits.open(path, memmap=False) as hd:
            data = hd[0].data
            hdr = hd[0].header

        if data.ndim == 2 and hdr.get("BAYERPAT"):
            # Simule le débayering pour obtenir les vraies dimensions
            h, w = data.shape
            c = 3
        else:
            data = to_hwc(data, hdr)
            h, w = data.shape[:2]
            c = data.shape[2] if data.ndim == 3 else 1

        logger.debug("get_image_shape(%s) -> %s", path, (h, w, c))
        return h, w, c

    else:
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data is None:
            raise RuntimeError(f"Failed to read {path}")

    shape = data.shape

    if data.ndim == 2:
        h, w = shape
        c = 1
    else:
        h, w, c = shape
    logger.debug("get_image_shape(%s) -> %s", path, (h, w, c))
    return h, w, c


def open_aligned_slice(path, y0, y1, wcs, wcs_ref, shape_ref, *, use_solver=True):
    """Return RGB slice (``y0:y1``) aligned to reference grid if possible."""

    ext = os.path.splitext(path)[1].lower()
    if ext in (".fit", ".fits", ".fts"):
        # Use ``memmap=True`` and disable automatic scaling so that only
        # the requested slice is read on demand.  ``memmap=False`` would
        # load the entire image into RAM.
        with fits.open(
            path,
            memmap=True,
            do_not_scale_image_data=True,
            ignore_blank=True,
        ) as hd:

            data = hd[0].data
            hdr = hd[0].header

        # Manually apply BSCALE/BZERO if present so the resulting array is
        # equivalent to ``hd[0].data`` with scaling enabled.
        bscale = hdr.get("BSCALE", 1.0)
        bzero = hdr.get("BZERO", 0.0)
        data = data.astype(np.float32, copy=False) * bscale + bzero

        if data.ndim == 2:
            bayer = hdr.get("BAYERPAT", "RGGB")  # Par défaut si absent
            try:
                if bayer.upper() in {"RGGB", "GRBG", "GBRG", "BGGR"}:
                    code = getattr(cv2, f"COLOR_Bayer{bayer.upper()}2RGB_EA", cv2.COLOR_BayerRG2RGB_EA)
                else:
                    code = cv2.COLOR_BayerRG2RGB_EA
                data = cv2.cvtColor(data.astype(np.uint16), code)
                logger.debug("Debayering image %s using pattern: %s", path, bayer)
            except Exception as e:
                logger.warning("Could not debayer %s: %s", path, e)
        else:
            data = to_hwc(data)

    else:
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).astype(np.float32)

    if use_solver:
        try:
            from seestar.core.alignment import SeestarAligner

            global aligner
            if aligner and hasattr(aligner, "reference_image_data") and aligner.reference_image_data is not None:
                aligned, ok = aligner._align_image(data, aligner.reference_image_data, os.path.basename(path))
                if ok:
                    data = aligned
                else:
                    _safe_print(f"⚠️ Alignement échoué pour {path}")
        except Exception as e:
            _safe_print(f"❌ Erreur alignement local: {e}")

    slice_data = data[y0:y1].copy()
    if "aligned" in locals():
        del aligned
    del data
    gc.collect()
    return slice_data


def _read_image(path: str) -> tuple[np.ndarray, fits.Header | None]:
    """Return image array and FITS header if applicable."""

    ext = os.path.splitext(path)[1].lower()
    if ext in {".fit", ".fits", ".fts"}:
        with fits.open(
            path,
            memmap=False,
            do_not_scale_image_data=True,
            ignore_blank=True,
        ) as hd:
            data = hd[0].data
            hdr = hd[0].header
        bscale = hdr.get("BSCALE", 1.0)
        bzero = hdr.get("BZERO", 0.0)
        data = data.astype(np.float32, copy=False) * bscale + bzero
        if data.ndim == 2:
            bayer = hdr.get("BAYERPAT", "RGGB")
            try:
                if bayer.upper() in {"RGGB", "GRBG", "GBRG", "BGGR"}:
                    code = getattr(cv2, f"COLOR_Bayer{bayer.upper()}2RGB_EA", cv2.COLOR_BayerRG2RGB_EA)
                else:
                    code = cv2.COLOR_BayerRG2RGB_EA
                data = cv2.cvtColor(data.astype(np.uint16), code)
            except Exception:
                logger.warning("Could not debayer %s", path)
        else:
            data = to_hwc(data, hdr)
        return data.astype(np.float32), hdr

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return img, None


def winsorize(tile, kappa, limit):
    """Winsorize ``tile`` in-place and return a floating point view.

    ``numpy.clip`` requires the output array to have a dtype compatible
    with the clipping bounds.  When a debayered image is fed in, ``tile``
    can be ``uint16`` which causes ``np.clip`` to raise a casting error
    when the calculated bounds are floating point.  Cast to ``float32``
    first so clipping succeeds and subsequent calculations operate on
    floating point data.
    """

    tile = tile.astype(np.float32, copy=False)
    med = np.median(tile, axis=0, keepdims=True)
    mad = np.median(np.abs(tile - med), axis=0, keepdims=True) * 1.4826
    lo = med - kappa * mad
    hi = med + kappa * mad
    np.clip(tile, lo, hi, out=tile)
    return tile


def _calc_snr(img: np.ndarray) -> float:
    if img.ndim == 3 and img.shape[2] == 3:
        data = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    else:
        data = img
    finite = data[np.isfinite(data)]
    if finite.size < 50:
        return 0.0
    signal = np.median(finite)
    mad = np.median(np.abs(finite - signal))
    noise = max(mad * 1.4826, 1e-9)
    return float(np.clip(signal / noise, 0.0, 1e9))


def _calc_star_score(img: np.ndarray) -> float:
    try:
        import astroalign as aa

        _t, (src, _dst) = aa.find_transform(img, img)
        num = len(src)
    except Exception:
        num = 0
    max_stars = 200.0
    return float(np.clip(num / max_stars, 0.0, 1.0))


def _compute_norm_params(images: list[np.ndarray], method: str):
    if not images or method == "none":
        return {}

    params = {}
    ref = images[0].astype(np.float32, copy=False)

    if method == "linear_fit":
        axis = (0, 1) if ref.ndim == 3 else None
        ref_low = np.nanpercentile(ref, 25.0, axis=axis)
        ref_high = np.nanpercentile(ref, 90.0, axis=axis)
        for i, img in enumerate(images):
            data = img.astype(np.float32, copy=False)
            low = np.nanpercentile(data, 25.0, axis=axis)
            high = np.nanpercentile(data, 90.0, axis=axis)
            d_src = high - low
            d_ref = ref_high - ref_low
            a = np.where(d_src > 1e-5, d_ref / np.maximum(d_src, 1e-9), 1.0)
            b = ref_low - a * low
            params[i] = (a.astype(np.float32), b.astype(np.float32))
    elif method == "sky_mean":

        def sky_val(im: np.ndarray) -> float:
            if im.ndim == 3 and im.shape[2] == 3:
                lum = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
            else:
                lum = im
            return float(np.nanpercentile(lum, 25.0))

        ref_sky = sky_val(ref)
        for i, img in enumerate(images):
            offset = ref_sky - sky_val(img.astype(np.float32, copy=False))
            params[i] = float(offset)

    return params


def flush_mmap(mmap_obj):
    mmap_obj.flush()
    try:
        MADV_DONTNEED = 4
        libc = ctypes.CDLL(None)
        libc.madvise(
            ctypes.c_void_p(int(mmap_obj.ctypes.data)),
            mmap_obj.nbytes,
            MADV_DONTNEED,
        )
    except Exception:
        pass


def classic_stack(
    csv_path: str,
    *,
    norm_method: str = "linear_fit",
    weight_method: str = "none",
    reject_algo: str = "none",
    kappa: float = 3.0,
    winsor: float = 0.05,
    api_key: str | None = None,
    use_solver: bool = True,
    correct_hot_pixels: bool = True,
    hot_threshold: float = 3.0,
    hot_neighborhood: int = 5,
    use_weighting: bool = False,
    snr_exp: float = 1.0,
    stars_exp: float = 0.5,
    min_weight: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack images sequentially like a classic stacker."""

    global aligner

    rows = read_rows(csv_path)
    if not rows:
        raise RuntimeError("CSV is empty")

    aligner = SeestarAligner()
    aligner.correct_hot_pixels = correct_hot_pixels
    aligner.hot_pixel_threshold = hot_threshold
    aligner.neighborhood_size = hot_neighborhood

    input_folder = os.path.dirname(rows[0]["path"])
    cand_files = [os.path.basename(r["path"]) for r in rows[:20]]
    tmp_dir = os.path.join(input_folder, "_temp_align_ref")
    os.makedirs(tmp_dir, exist_ok=True)
    ref_img, ref_hdr = aligner._get_reference_image(input_folder, cand_files, tmp_dir)
    if ref_img is None:
        raise RuntimeError("Failed to select reference image")
    aligner.reference_image_data = ref_img
    logger.info("Reference selected: %s", ref_hdr.get("HIERARCH SEESTAR REF SRCFILE", "auto"))

    shape_ref = ref_img.shape[:2]
    H, W = shape_ref
    C = ref_img.shape[2] if ref_img.ndim == 3 else 1

    wcs_ref = None
    if ref_hdr is not None:
        try:
            wcs_ref = WCS(ref_hdr)
        except Exception:
            wcs_ref = None

    axis = (0, 1) if ref_img.ndim == 3 else None
    ref_low = np.nanpercentile(ref_img, 25.0, axis=axis)
    ref_high = np.nanpercentile(ref_img, 90.0, axis=axis)

    def sky_val(im: np.ndarray) -> float:
        if im.ndim == 3 and im.shape[2] == 3:
            lum = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
        else:
            lum = im
        return float(np.nanpercentile(lum, 25.0))

    ref_sky = sky_val(ref_img)

    images: list[np.ndarray] = []
    weights: list[float] = []
    wcs_cache: dict[str, WCS | None] = {}

    for idx, row in enumerate(rows, 1):
        path = row["path"]
        img, hdr = _read_image(path)

        if correct_hot_pixels:
            try:
                from seestar.core.hot_pixels import detect_and_correct_hot_pixels

                img = detect_and_correct_hot_pixels(img, hot_threshold, hot_neighborhood)
            except Exception:
                pass

        w = 1.0
        if use_weighting:
            if weight_method == "snr":
                w = _calc_snr(img) ** snr_exp
            elif weight_method == "stars":
                w = _calc_star_score(img) ** stars_exp
        w *= float(row.get("weight") or 1.0)
        w = max(float(w), min_weight, 1e-9)

        if norm_method == "linear_fit":
            data = img.astype(np.float32, copy=False)
            low = np.nanpercentile(data, 25.0, axis=axis)
            high = np.nanpercentile(data, 90.0, axis=axis)
            d_src = high - low
            d_ref = ref_high - ref_low
            a = np.where(d_src > 1e-5, d_ref / np.maximum(d_src, 1e-9), 1.0)
            b = ref_low - a * low
            img = data * a + b
        elif norm_method == "sky_mean":
            img = img.astype(np.float32) + (ref_sky - sky_val(img.astype(np.float32)))
        else:
            img = img.astype(np.float32)

        aligned, ok = aligner._align_image(img, aligner.reference_image_data, os.path.basename(path))
        if not ok and use_solver:
            if path not in wcs_cache:
                wcs = solve_local_plate(path)
                if wcs is None:
                    wcs = get_wcs_from_astap(path)
                if wcs is None:
                    wcs = solve_with_astrometry_local(path)
                if wcs is None and api_key:
                    wcs = solve_with_astrometry_net(path, api_key)
                wcs_cache[path] = wcs
            wcs = wcs_cache.get(path)
            if wcs is not None and wcs_ref is not None:
                try:
                    aligned = warp_image(img, wcs, wcs_ref, shape_ref)
                    ok = True
                except Exception as e:
                    logger.warning("WCS align failed for %s: %s", path, e)
        if not ok:
            logger.warning("Alignment failed for %s", path)
        images.append(aligned.astype(np.float32))
        weights.append(w)
        logger.info("Loaded %d/%d: %s", idx, len(rows), os.path.basename(path))

    weights_arr = np.asarray(weights, dtype=np.float32)

    if reject_algo == "winsorized_sigma":
        final, _ = _stack_winsorized_sigma(images, weights_arr, kappa=kappa, winsor_limits=(winsor, winsor))
    elif reject_algo == "kappa_sigma":
        final, _ = _stack_kappa_sigma(images, weights_arr, sigma_low=kappa, sigma_high=kappa)
    else:
        final, _ = _stack_mean(images, weights_arr)

    weight_map = np.full((H, W), float(np.sum(weights_arr)), dtype=np.float32)

    bg = np.median(final) if np.isfinite(final).any() else 0.0
    final = final - bg

    return final.astype(np.float32), weight_map


def stream_stack(
    csv_path,
    out_sum,
    out_wht,
    *,
    tile=512,
    kappa=3.0,
    winsor=0.05,
    api_key=None,
    use_solver=True,
    max_mem=None,
    norm_method="none",
    weight_method="none",
    reject_algo="winsorized_sigma",
    correct_hot_pixels=True,
    hot_threshold=3.0,
    hot_neighborhood=5,
    use_weighting=False,
    snr_exp=1.0,
    stars_exp=0.5,
    min_weight=0.1,
    progress_callback=None,
    cleanup_temp_files=True,
):
    global aligner
    rows = read_rows(csv_path)
    aligner = SeestarAligner()
    aligner.correct_hot_pixels = correct_hot_pixels
    aligner.hot_pixel_threshold = hot_threshold
    aligner.neighborhood_size = hot_neighborhood
    input_folder = os.path.dirname(rows[0]["path"])
    files_to_scan = [os.path.basename(row["path"]) for row in rows]
    tmp_output_dir = os.path.join(input_folder, "_temp_align_ref")

    os.makedirs(tmp_output_dir, exist_ok=True)

    ref_img, _ = aligner._get_reference_image(input_folder, files_to_scan, tmp_output_dir)
    if ref_img is not None:
        aligner.reference_image_data = ref_img
        _safe_print("✅ Image de référence chargée pour alignement local")
    else:

        aligner = SeestarAligner()
    if not rows:
        raise RuntimeError("CSV is empty")

    logger.info("Début du traitement")

    stream_stack._start_time = time.monotonic()

    stream_stack._next_pct = 0.0

    if max_mem is None:
        max_mem_bytes = int(os.getenv("SEESTAR_MAX_MEM", 2_000_000_000))
    else:
        max_mem_bytes = int(float(max_mem) * 1024**3)

    first = rows[0]["path"]
    H, W, C = get_image_shape(first)
    shape_ref = (H, W)
    logger.debug(
        "stream_stack: %d files, first=%s, shape=%dx%dx%d",
        len(rows),
        first,
        H,
        W,
        C,
    )

    wcs_cache: dict[str, object] = {}

    wcs_ref: WCS | None = None

    if use_solver:
        for i, row in enumerate(rows, 1):
            path = row["path"]
            if path in wcs_cache:
                continue
            method = "local"
            wcs = solve_local_plate(path)
            if wcs is None:
                wcs = get_wcs_from_astap(path)
                if wcs is not None:
                    method = "astap"
            if wcs is None:
                wcs = solve_with_astrometry_local(path)
                if wcs is not None:
                    method = "astrometry_local"
            if wcs is None and api_key:
                try:
                    wcs = solve_with_astrometry_net(path, api_key)
                    if wcs is not None:
                        method = "astrometry_net"
                except Exception:
                    wcs = None

            if wcs is None:
                logger.warning("Plate-solve failed for %s", path)
            else:
                if wcs_ref is None:
                    wcs_ref = wcs
                _safe_print(f"Solved {i}/{len(rows)}: {os.path.basename(path)} via {method}")
            wcs_cache[path] = wcs

        if wcs_ref is not None:
            _safe_print("ALIGN OK")

        if wcs_ref is None:
            logger.warning("Reference WCS not resolved; stacking without alignment")
    else:
        for row in rows:
            wcs_cache[row["path"]] = None

    aligned_dir = os.path.join(input_folder, "_aligned_tmp")
    os.makedirs(aligned_dir, exist_ok=True)

    # Align each image once and save it for later
    norm_params = {}
    weights_scalar: list[float] = []
    aligned_paths: list[str] = []
    ref_low = ref_high = None
    ref_sky = None
    for idx, row in enumerate(rows):
        img = open_aligned_slice(
            row["path"],
            0,
            H,
            wcs_cache[row["path"]],
            wcs_ref,
            shape_ref,
            use_solver=use_solver,
        )
        if correct_hot_pixels:
            try:
                from seestar.core.hot_pixels import detect_and_correct_hot_pixels

                img = detect_and_correct_hot_pixels(img, hot_threshold, hot_neighborhood)
            except Exception:
                pass
        if norm_method == "linear_fit":
            data = img.astype(np.float32, copy=False)
            axis = (0, 1) if data.ndim == 3 else None
            low = np.nanpercentile(data, 25.0, axis=axis)
            high = np.nanpercentile(data, 90.0, axis=axis)
            if idx == 0:
                ref_low = low
                ref_high = high
                norm_params[idx] = (np.ones_like(low, dtype=np.float32), np.zeros_like(low, dtype=np.float32))
            else:
                d_src = high - low
                d_ref = ref_high - ref_low
                a = np.where(d_src > 1e-5, d_ref / np.maximum(d_src, 1e-9), 1.0)
                b = ref_low - a * low
                norm_params[idx] = (a.astype(np.float32), b.astype(np.float32))
        elif norm_method == "sky_mean":

            def sky_val(im: np.ndarray) -> float:
                if im.ndim == 3 and im.shape[2] == 3:
                    lum = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
                else:
                    lum = im
                return float(np.nanpercentile(lum, 25.0))

            offset = sky_val(img.astype(np.float32, copy=False))
            if idx == 0:
                ref_sky = offset
                norm_params[idx] = 0.0
            else:
                norm_params[idx] = ref_sky - offset

        w = 1.0
        if use_weighting:
            if weight_method == "snr":
                w = _calc_snr(img) ** snr_exp
            elif weight_method == "stars":
                w = _calc_star_score(img) ** stars_exp
        w = max(float(w), min_weight, 1e-9)
        weights_scalar.append(w)

        # Save aligned image for later stacking
        temp_path = os.path.join(aligned_dir, f"aligned_{idx:04d}.npy")
        np.save(temp_path, img.astype(np.float32))
        aligned_paths.append(temp_path)

        # Release the aligned image to keep memory usage low
        del img
        gc.collect()

    out_sum = os.path.abspath(str(out_sum)).strip()
    out_wht = os.path.abspath(str(out_wht)).strip()
    cum_sum = open_memmap(out_sum, "w+", dtype=np.float32, shape=(H, W, C))
    cum_sum[:] = 0
    cum_wht = open_memmap(out_wht, "w+", dtype=np.float32, shape=(H, W))
    cum_wht[:] = 1
    logger.debug("allocated accumulators: cum_sum %s, cum_wht %s", cum_sum.shape, cum_wht.shape)

    tile_h = int(tile)
    image_count = 0
    for y0 in range(0, H, tile_h):
        y1 = min(y0 + tile_h, H)
        rows_h = y1 - y0
        logger.debug(f"RAM avant la tuile {y0}-{y1} : {psutil.virtual_memory().used / 1024**2:.2f} MB")
        per_img_bytes = rows_h * W * C * 4
        group_size = max(1, max_mem_bytes // max(per_img_bytes, 1))

        tile_sum = np.zeros((rows_h, W, C), dtype=np.float32)
        tile_wht = 0.0
        # Load aligned images from disk in manageable batches
        for s in range(0, len(aligned_paths), group_size):
            batch_files = aligned_paths[s : s + group_size]
            batch_imgs = np.empty((len(batch_files), rows_h, W, C), dtype=np.float32)
            weights_arr = np.empty(len(batch_files), dtype=np.float32)
            for j, (idx, p) in enumerate(zip(range(s, s + len(batch_files)), batch_files)):
                arr = np.load(p, mmap_mode="r")
                img_slice = arr[y0:y1]
                if correct_hot_pixels:
                    try:
                        from seestar.core.hot_pixels import detect_and_correct_hot_pixels
                        img_slice = detect_and_correct_hot_pixels(img_slice, hot_threshold, hot_neighborhood)
                    except Exception:
                        pass
                weight = float(rows[idx].get("weight") or 1.0) * weights_scalar[idx]
                if img_slice.ndim == 2:
                    img_slice = img_slice[..., None]
                if img_slice.shape[2] != C:
                    if img_slice.shape[2] == 3 and C == 1:
                        img_slice = cv2.cvtColor(img_slice, cv2.COLOR_RGB2GRAY)[..., None]
                    elif img_slice.shape[2] == 1 and C == 3:
                        img_slice = np.repeat(img_slice, 3, axis=2)
                    else:
                        raise ValueError(f"Image channel mismatch: expected {C}, got {img_slice.shape[2]}")
                if norm_method == "linear_fit" and idx in norm_params:
                    a, b = norm_params[idx]
                    img_slice = img_slice.astype(np.float32) * a + b
                elif norm_method == "sky_mean" and idx in norm_params:
                    img_slice = img_slice.astype(np.float32) + norm_params[idx]
                if reject_algo == "winsorized_sigma":
                    img_slice = winsorize(img_slice, kappa, winsor)
                batch_imgs[j] = img_slice
                weights_arr[j] = weight
                del img_slice  # FIX MEMLEAK
                gc.collect()  # FIX MEMLEAK
                image_count += 1
                if image_count % 100 == 0:
                    vm = psutil.virtual_memory()
                    ram_mb = vm.used / (1024**2)
                    cache_mb = getattr(vm, "cached", 0.0) / (1024**2)
                    logger.info(
                        "Loaded %d images | RAM %.1f MB | Cache %.1f MB",
                        image_count,
                        ram_mb,
                        cache_mb,
                    )
                    _safe_print(
                        f"{image_count} images loaded | RAM {ram_mb:.1f}MB | Cache {cache_mb:.1f}MB",
                        flush=True,
                    )

            if reject_algo == "winsorized_sigma":
                stacked_tile, _ = _stack_winsorized_sigma(
                    batch_imgs,
                    weights_arr,
                    kappa=kappa,
                    winsor_limits=(winsor, winsor),
                    max_mem_bytes=max_mem_bytes,
                )
            elif reject_algo == "kappa_sigma":
                stacked_tile, _ = _stack_kappa_sigma(
                    batch_imgs,
                    weights_arr,
                    sigma_low=kappa,
                    sigma_high=kappa,
                )
            else:
                stacked_tile, _ = _stack_mean(
                    batch_imgs,
                    weights_arr,
                )
            # FIX MEMLEAK: clean stacked_tile immediately
            weight_sum = float(np.sum(weights_arr))
            tile_sum += stacked_tile.astype(np.float32) * weight_sum
            tile_wht += weight_sum
            del stacked_tile  # FIX MEMLEAK
            del batch_imgs, weights_arr  # FIX MEMLEAK
            gc.collect()  # FIX MEMLEAK

        cum_sum[y0:y1] = tile_sum
        cum_wht[y0:y1] = (
            tile_wht if isinstance(tile_wht, np.ndarray) else np.full((rows_h, W), float(tile_wht), dtype=np.float32)
        )
        flush_mmap(cum_sum)
        flush_mmap(cum_wht)
        gc.collect()
        logger.debug(f"RAM après la tuile {y0}-{y1} : {psutil.virtual_memory().used / 1024**2:.2f} MB")
        if y0 == 0:
            logger.debug("stacked first tile -> cum_sum slice %s", cum_sum[y0:y1].shape)
        progress = 100.0 * y1 / H
        if progress >= stream_stack._next_pct or y1 == H:
            elapsed = time.monotonic() - stream_stack._start_time
            frac = y1 / H
            eta = (elapsed / frac) * (1 - frac) if frac > 0 else 0.0
            swap_used_mb = psutil.swap_memory().used / (1024**2)
            eta_str = _format_seconds(eta)
            logger.info(
                "Progress %5.1f%% | ETA %s | Swap %.1f MB",
                progress,
                eta_str,
                swap_used_mb,
            )
            # Ligne de texte simple pour la GUI principale (se termine par '%').
            _safe_print(f"{progress:.1f}%", flush=True)
            if progress_callback:
                try:
                    # Mise à jour de la barre de progression
                    progress_callback(f"{progress:.1f}%", progress)
                    # Mise à jour de l'ETA dans la GUI
                    progress_callback(f"ETA_UPDATE:{eta_str}", None)
                except Exception:
                    pass
            stream_stack._next_pct = min(stream_stack._next_pct + 10.0, 100.0)

    if cleanup_temp_files:
        shutil.rmtree(aligned_dir, ignore_errors=True)

    vm_end = psutil.virtual_memory()
    ram_end_mb = vm_end.used / (1024**2)
    cache_end_mb = getattr(vm_end, "cached", 0.0) / (1024**2)
    logger.info(
        "Final RAM %.1f MB | Cache %.1f MB",
        ram_end_mb,
        cache_end_mb,
    )
    _safe_print(
        f"Final RAM {ram_end_mb:.1f}MB | Cache {cache_end_mb:.1f}MB",
        flush=True,
    )

    return cum_sum, cum_wht


def main():
    args = parse_args()

    if args.weight in {"noise_variance", "quality"}:
        args.weight = "snr"
    elif args.weight == "noise_fwhm":
        args.weight = "stars"

    #    if args.batch_size == 1:
    #        args.use_solver = False

    os.makedirs(args.out, exist_ok=True)

    final, weight_map = classic_stack(
        args.csv,
        norm_method=args.norm,
        weight_method=args.weight,
        reject_algo=args.reject,
        kappa=args.kappa,
        winsor=args.winsor,
        api_key=args.api_key,
        use_solver=args.use_solver,
        correct_hot_pixels=args.correct_hot_pixels,
        hot_threshold=args.hot_threshold,
        hot_neighborhood=args.hot_neighborhood,
        use_weighting=args.use_weighting,
        snr_exp=args.snr_exp,
        stars_exp=args.stars_exp,
        min_weight=args.min_weight,
    )

    logger.debug("final image shape before squeeze: %s", final.shape)

    # --- Chromatic balancing step (similar to queue_manager) ---
    if final.ndim == 3 and final.shape[2] == 3:
        try:
            from seestar.enhancement.color_correction import ChromaticBalancer

            chroma = ChromaticBalancer(border_size=25, blur_radius=8)
            max_val = float(np.nanmax(final))
            if max_val > 0:
                scaled = np.clip(final / max_val, 0.0, 1.0)
                corrected = chroma.normalize_stack(scaled)
                final = np.clip(corrected, 0.0, 1.0) * max_val
            else:
                final = chroma.normalize_stack(np.clip(final, 0.0, 1.0))
        except Exception as e:
            logger.warning("ChromaticBalancer failed: %s", e)

    if final.ndim == 3 and final.shape[2] == 3:
        # Write colour data in (C, H, W) order so each channel is stored as a
        # separate FITS plane.  This preserves the RGB information instead of
        # swapping the height and width axes which produced a greyscale image
        # in some FITS viewers.
        fits_data = final.transpose(2, 0, 1)
    elif final.ndim == 3 and final.shape[2] == 1:
        # Single channel image: duplicate the plane so that FITS viewers treat
        # the output as RGB.  Each duplicated channel is stored in its own
        # plane to remain consistent with the RGB case above.
        fits_data = np.repeat(final, 3, axis=2).transpose(2, 0, 1)
    else:
        fits_data = final.squeeze()

    fits.writeto(
        os.path.join(args.out, "final.fits"),
        fits_data.astype(np.float32),
        overwrite=True,
    )
    preview = np.clip(final, 0, 1) ** 0.5
    preview = np.clip(preview * 255, 0, 255).astype(np.uint8)
    if preview.ndim == 3 and preview.shape[2] == 1:
        preview = preview[:, :, 0]
    imageio.imwrite(os.path.join(args.out, "preview.png"), preview)

    weight_path = os.path.join(args.out, "weight_map.npy")
    np.save(weight_path, weight_map.astype(np.float32))

    return 0


if __name__ == "__main__":
    if os.getenv("BORING_TEST"):
        import tempfile
        import shutil

        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(message)s")

        tmp = tempfile.mkdtemp()
        fits.writeto(
            os.path.join(tmp, "c_hw.fits"),
            (np.arange(60, dtype=np.uint16).reshape(3, 4, 5)),
            overwrite=True,
        )
        csv_path = os.path.join(tmp, "plan.csv")
        with open(csv_path, "w") as f:
            f.write("file_path\n" + tmp + "/c_hw.fits\n")
        img, wht = classic_stack(csv_path)
        assert img.shape == (4, 5, 3), img.shape
        shutil.rmtree(tmp)
        sys.exit(0)

    try:
        sys.exit(main())
    except Exception:
        logging.exception("Fatal error in boring_stack")
        sys.exit(1)
