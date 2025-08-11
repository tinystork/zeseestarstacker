import argparse
import csv
import os
import sys
import time
import logging
import shutil
import tempfile
import gc
import psutil
import signal
import numpy as np
import json
from typing import Optional




_PROC = psutil.Process(os.getpid())  # Track process RAM for DEBUG logs

# Global reference to the running stacker for signal handlers
_GLOBAL_STACKER = None


def _handle_signal(signum, frame):
    """Gracefully stop the stacker when a termination signal is received."""
    global _GLOBAL_STACKER
    try:
        if _GLOBAL_STACKER is not None and _GLOBAL_STACKER.is_running():
            _GLOBAL_STACKER.stop()
    except Exception:
        pass

def _log_mem(tag: str) -> None:
    """Log RSS memory to help locate leaks."""
    try:
        rss = _PROC.memory_info().rss / (1024 * 1024)
        logger.debug("RAM usage [%s]: %.1f MB", tag, rss)
    except Exception:
        pass

def _cleanup_stacker(stacker):
    """Free all heavy resources held by a SeestarQueuedStacker instance."""
    if stacker is None:
        return
    # 1. finish async drizzle jobs
    try:
        stacker._wait_drizzle_processes()
    except Exception:
        pass
    # 2. shutdown executors
    for exe_name in ("drizzle_executor", "quality_executor"):
        exe = getattr(stacker, exe_name, None)
        if exe:
            exe.shutdown(wait=True, cancel_futures=True)
    # 3. flush / close memmaps and drop references
    for arr_name in ("cumulative_sum_memmap", "cumulative_wht_memmap"):
        arr = getattr(stacker, arr_name, None)
        if arr is not None:
            try:
                arr.flush()
                if hasattr(arr, "_mmap") and arr._mmap is not None:
                    arr._mmap.close()
            except Exception:
                pass
            finally:
                setattr(stacker, arr_name, None)
                gc.collect()
    # 4. close any remaining memmaps via the stacker helper
    if hasattr(stacker, "_close_memmaps"):
        try:
            stacker._close_memmaps()
        except Exception:
            pass
        finally:
            gc.collect()
    # 5. delete memmap files if requested
    if getattr(stacker, "perform_cleanup", False):
        for path_name in ("cumulative_sum_path", "cumulative_wht_path"):
            p = getattr(stacker, path_name, None)
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
    # 6. clear caches & run GC
    getattr(stacker, "_indices_cache", {}).clear()
    gc.collect()
    _log_mem("cleanup")

from numpy.lib.format import open_memmap as _orig_open_memmap
import weakref


def _safe_open_memmap(filename, *args, **kwargs):
    """Fallback to a temp directory if the original path fails.

    ``gc.collect`` is invoked after creation to free any residual buffers.
    """
    try:
        mm = _orig_open_memmap(filename, *args, **kwargs)
    except OSError:
        tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(str(filename)))
        mm = _orig_open_memmap(tmp_path, *args, **kwargs)
    finally:
        gc.collect()
        _log_mem(f"memmap_open:{os.path.basename(str(filename))}")
    try:
        if hasattr(mm, "_mmap") and mm._mmap is not None:
            weakref.finalize(mm, mm._mmap.close)
    except Exception:
        pass
    return mm

# Ensure UTF-8 console for progress messages
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# When executed directly from source, make project root importable
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from seestar.gui.settings import SettingsManager

import numpy.lib.format as _np_format
_np_format.open_memmap = _safe_open_memmap

from seestar.queuep.queue_manager import SeestarQueuedStacker
from seestar.core.alignment import SeestarAligner
from seestar.core.normalization import (
    _normalize_images_linear_fit,
    _normalize_images_sky_mean,
)
from seestar.core.streaming_stack import stack_disk_streaming
from seestar.queuep.queue_manager import _quality_metrics_worker
# Import helpers from the core package.  Use an absolute import so that this
# script can be executed directly without Python considering it a package
# relative module.  When ``__package__`` is ``None`` the project root is added
# to ``sys.path`` above, allowing this import to succeed.
from seestar.core.image_processing import (
    load_and_validate_fits,
    save_fits_image,
    sanitize_header_for_wcs,
)
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# ASTAP WCS helpers
# -----------------------------------------------------------------------------


def _sanitize_astap_wcs(path: str) -> None:
    """Ensure CONTINUE card values in ASTAP ``.wcs`` sidecars are strings.

    ``path`` may point either to the original FITS file or directly to the
    ``.wcs`` sidecar produced by ASTAP.
    """
    wcs_path = path if path.lower().endswith(".wcs") else os.path.splitext(path)[0] + ".wcs"
    if not os.path.exists(wcs_path):
        return
    try:
        # Primary attempt: treat the sidecar as a minimal FITS file
        with fits.open(wcs_path, mode="update") as hdul:  # pragma: no cover - best effort

            hdr = hdul[0].header
            modified = False
            for card in hdr.cards:
                if card.keyword == "CONTINUE" and not isinstance(card.value, str):
                    card.value = str(card.value)
                    modified = True
            if modified:
                hdul.flush()

            return
    except Exception:
        pass
    # Fallback: some ASTAP ``.wcs`` files are plain text headers
    try:  # pragma: no cover - best effort
        header = fits.Header.fromfile(
            wcs_path, sep="\n", padding=False, endcard=False
        )
        modified = False
        for card in header.cards:
            if card.keyword == "CONTINUE" and not isinstance(card.value, str):
                card.value = str(card.value)
                modified = True
        if modified:
            with open(wcs_path, "w", newline="\n") as f:
                f.write(header.tostring(sep="\n"))
    except Exception:

        logger.debug("Échec de la correction CONTINUE pour %s", wcs_path, exc_info=True)


# -----------------------------------------------------------------------------
# CSV helpers (reused by GUI)
# -----------------------------------------------------------------------------

def read_rows(csv_path):
    """Read rows from ``csv_path`` with optional headers."""
    logger.info("lecture de stack_plan.csv: %s", csv_path)
    rows_out = []
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
        has_header = any(
            h in {"order", "file", "filename", "path", "index", "weight"} for h in header
        )
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
    return [r["path"] for r in read_rows(csv_path)]


# -----------------------------------------------------------------------------
# Reproject+Coadd helper
# -----------------------------------------------------------------------------

def _finalize_reproject_and_coadd(aligned_files, reference_path, output_path):
    """Reproject ``aligned_files`` onto the reference grid and combine."""

    from seestar import reproject_utils
    try:
        ref_hdr = fits.getheader(reference_path, memmap=False)
    except Exception:
        ref_hdr = fits.Header()
    sanitize_header_for_wcs(ref_hdr)
    ref_wcs = WCS(ref_hdr, naxis=2)
    try:
        result, _ = reproject_utils.reproject_and_coadd_from_paths(
            aligned_files,
            output_projection=ref_wcs,
            reproject_function=reproject_utils.reproject_interp,
            match_background=True,
        )
    except Exception:
        logger.exception("reproject_and_coadd failed")
        return False

    fits.PrimaryHDU(
        data=np.moveaxis(result, -1, 0) if result.ndim == 3 else result,
        header=ref_wcs.to_header(relax=True),
    ).writeto(output_path, overwrite=True)
    return True


# -----------------------------------------------------------------------------
# Global normalization helpers
# -----------------------------------------------------------------------------

def _calculate_global_linear_fit(reference_path, paths, low=25.0, high=90.0):
    """Return per-image slope/intercept using a common reference image."""
    ref_data, _ = load_and_validate_fits(reference_path)
    if ref_data is None:
        return [(1.0, 0.0) for _ in paths]
    ref = ref_data.astype(np.float32, copy=False)
    is_color = ref.ndim == 3 and ref.shape[-1] == 3
    ref_low = np.nanpercentile(ref, low, axis=(0, 1) if is_color else None)
    ref_high = np.nanpercentile(ref, high, axis=(0, 1) if is_color else None)
    delta_ref = ref_high - ref_low
    coeffs = []
    for p in paths:
        data, _ = load_and_validate_fits(p)
        if data is None:
            coeffs.append((1.0, 0.0))
            continue
        img = data.astype(np.float32, copy=False)
        low_p = np.nanpercentile(img, low, axis=(0, 1) if is_color else None)
        high_p = np.nanpercentile(img, high, axis=(0, 1) if is_color else None)
        delta_src = high_p - low_p
        a = np.where(delta_src > 1e-5, delta_ref / np.maximum(delta_src, 1e-9), 1.0)
        b = ref_low - a * low_p
        coeffs.append((a, b))
    return coeffs


def _calculate_global_sky_mean(paths, sky_percentile=25.0):
    """Return the average sky value from all images."""
    vals = []
    for p in paths:
        data, _ = load_and_validate_fits(p)
        if data is None:
            continue
        if data.ndim == 3 and data.shape[-1] == 3:
            lum = 0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]
        else:
            lum = data
        vals.append(np.nanpercentile(lum, sky_percentile))
    if not vals:
        return 0.0
    return float(np.nanmean(vals))


def _apply_linear_fit(data, slope, intercept):
    return slope * data + intercept


def _apply_sky_mean(data, target_sky, sky_percentile=25.0):
    if data.ndim == 3 and data.shape[-1] == 3:
        lum = 0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]
    else:
        lum = data
    curr = np.nanpercentile(lum, sky_percentile)
    return data + (target_sky - curr)


# -----------------------------------------------------------------------------
# batch_size=1 helpers
# -----------------------------------------------------------------------------


def _apply_normalization_and_weight(
    aligned_img: np.ndarray,
    ref_img: np.ndarray,
    norm_method: str,
    weight_method: str,
    snr_exp: float,
    stars_exp: float,
) -> tuple[np.ndarray, float]:
    """Normalize ``aligned_img`` against ``ref_img`` and compute a raw weight.

    The returned weight is *not* normalized across images; callers should
    renormalize the collection of weights once all images have been processed.
    """

    img = aligned_img.astype(np.float32, copy=False)
    ref = ref_img.astype(np.float32, copy=False)

    if norm_method == "linear_fit":
        img = _normalize_images_linear_fit([ref, img], reference_index=0)[1]
    elif norm_method == "sky_mean":
        img = _normalize_images_sky_mean([ref, img], reference_index=0)[1]

    scores, _, _ = _quality_metrics_worker(img)
    weight = 1.0
    if weight_method == "snr":
        weight = max(scores.get("snr", 0.0), 0.0) ** snr_exp
    elif weight_method == "stars":
        weight = max(scores.get("stars", 0.0), 0.0) ** stars_exp
    return img, max(weight, 1e-9)


def _normalize_weights(weights: list[float], min_weight: float) -> list[float]:
    arr = np.asarray(weights, dtype=np.float32)
    if arr.size == 0:
        return []
    sum_w = float(arr.sum())
    if sum_w > 1e-9:
        arr *= arr.size / sum_w
    else:
        arr[:] = 1.0
    arr = np.maximum(arr, float(min_weight))
    sum_w2 = float(arr.sum())
    if sum_w2 > 1e-9:
        arr *= arr.size / sum_w2
    else:
        arr[:] = 1.0
    return arr.tolist()


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Batch-1 stacking using QueueManager")
    # Weight options mirror the GUI: snr, stars, noise variance and noise+FWHM
    p.add_argument("--csv", required=True, help="CSV with file list")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--tile", type=int, default=512, help="Tile height (ignored)")
    p.add_argument("--kappa", type=float, default=3.0, help="Kappa value")
    p.add_argument("--winsor", type=float, default=0.05, help="Winsor limit")
    p.add_argument(
        "--max-mem",
        type=float,
        default=None,
        help="Maximum HQ memory in GB (overrides SEESTAR_MAX_MEM)",
    )
    p.add_argument("--api-key", default=None, help="Astrometry.net API key")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Flush intermediate results every N images when batch size is 1"
        ),
    )
    p.add_argument("--norm", default="none", choices=["linear_fit", "sky_mean", "none"], help="Normalization")
    p.add_argument(
        "--weight",
        default="none",
        choices=["snr", "stars", "noise_variance", "noise_fwhm", "none"],
        help="Weighting method",
    )
    p.add_argument("--reject", default="winsorized_sigma", choices=["kappa_sigma", "winsorized_sigma", "none"], help="Rejection algorithm")
    p.add_argument("--no-hot-pixels", dest="correct_hot_pixels", action="store_false")
    p.add_argument("--hot-threshold", type=float, default=3.0)
    p.add_argument("--hot-neighborhood", type=int, default=5)
    p.add_argument("--use-weighting", action="store_true", default=False)
    p.add_argument("--snr-exp", type=float, default=1.0)
    p.add_argument("--stars-exp", type=float, default=0.5)
    p.add_argument("--min-weight", type=float, default=0.1)
    p.add_argument("--use-solver", dest="use_solver", action="store_true")
    p.add_argument("--no-solver", dest="use_solver", action="store_false")
    p.add_argument("--cleanup-temp-files", dest="cleanup_temp_files", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--align-on-disk", action="store_true", help="Use memmap files during alignment")
    p.add_argument("--show-progress", action="store_true", help="Display a minimal progress GUI")
    p.add_argument(
        "--final-combine",
        choices=["none", "mean", "reject", "reproject", "reproject_coadd"],
        default=None,
        help="Override final combine strategy (CLI has priority over settings).",
    )
    p.set_defaults(use_solver=True)
    return p.parse_args()


def _resolve_final_combine(cli_value: Optional[str], settings) -> str:
    alias_map = {
        "reproject and coadd": "reproject_coadd",
        "reproject_and_coadd": "reproject_coadd",
        "reproject-coadd": "reproject_coadd",
        "reproject": "reproject",
        "mean": "mean",
        "reject": "reject",
        "none": "none",
        None: None,
        "": None,
    }

    def norm(v):
        if v is None:
            return None
        v = str(v).strip().lower()
        return alias_map.get(v, v)

    final_from_cli = norm(cli_value)
    final_from_settings = norm(getattr(settings, "stack_final_combine", None))

    if final_from_cli:
        source = "CLI"
        resolved = final_from_cli
    elif final_from_settings:
        source = "settings"
        resolved = final_from_settings
    else:
        source = "default"
        resolved = "mean"

    logging.info(f"[final-combine] resolved='{resolved}' (source={source})")
    return resolved


# -----------------------------------------------------------------------------
# Main processing logic
# -----------------------------------------------------------------------------

def _run_stack(args, progress_cb) -> int:
    """Execute the stacking process using ``progress_cb`` for updates."""
    rows = read_rows(args.csv)
    if not rows:
        logger.error("CSV is empty")
        return 1

    # Load user settings early to resolve final combine strategy
    settings = SettingsManager()
    try:
        settings.load_settings()
    except Exception:
        settings.reset_to_defaults()

    final_combine = _resolve_final_combine(getattr(args, "final_combine", None), settings)
    reproject_coadd_final = final_combine == "reproject_coadd"
    settings.stack_final_combine = final_combine
    settings.reproject_coadd_final = reproject_coadd_final
    logger.info(
        "Parsed final_combine=%s, batch_size=%s", final_combine, args.batch_size
    )

    if args.batch_size == 1 and final_combine not in ("reproject", "reproject_coadd"):
        logger.info(
            "=== MODE BORING STACK (batch_size=1, final_combine=%s) ===",
            final_combine,
        )
        all_paths = [r["path"] for r in rows]
        ref_path = getattr(args, "reference_image_path", "") or all_paths[0]
        ref_data, _ = load_and_validate_fits(ref_path)
        if ref_data is None:
            logger.error("Failed to load reference image %s", ref_path)
            return 1
        logger.info("Référence utilisée: %s", ref_path)

        aligned_dir = os.path.join(tempfile.gettempdir(), "aligned_tmp")
        os.makedirs(aligned_dir, exist_ok=True)

        aligner = SeestarAligner()
        aligned_paths: list[str] = []
        raw_weights: list[float] = []

        total = len(rows)
        for idx, row in enumerate(rows):
            img_path = row["path"]
            img_data, _ = load_and_validate_fits(img_path)
            if img_data is None:
                logger.warning("Failed to load %s", img_path)
                continue
            aligned_img, success = aligner._align_image(
                img_data,
                ref_data,
                os.path.basename(img_path),
                force_same_shape_as_ref=True,
                use_disk=args.align_on_disk,
            )
            if not success or aligned_img is None:
                logger.warning("Alignment failed for %s", img_path)
                continue

            norm_img, raw_w = _apply_normalization_and_weight(
                aligned_img,
                ref_data,
                args.norm,
                args.weight if args.use_weighting else "none",
                args.snr_exp,
                args.stars_exp,
            )
            raw_weights.append(raw_w)

            npy_path = os.path.join(aligned_dir, f"aligned_{idx:05d}.npy")
            if norm_img.ndim == 3 and norm_img.shape[2] == 1:
                to_save = norm_img[:, :, 0]
            else:
                to_save = norm_img
            np.save(npy_path, to_save.astype(np.float32))
            aligned_paths.append(npy_path)

            if progress_cb:
                progress_cb(f"Aligned: {idx + 1}", (idx + 1) / max(total, 1) * 100)

        weights = (
            _normalize_weights(raw_weights, args.min_weight)
            if args.use_weighting
            else None
        )

        mode_map = {
            "kappa_sigma": "kappa-sigma",
            "winsorized_sigma": "winsorized-sigma",
            "none": "mean",
        }
        stack_mode = mode_map.get(args.reject, "winsorized-sigma")

        final_path = stack_disk_streaming(
            aligned_paths,
            mode=stack_mode,
            weights=weights,
            chunk_rows=args.chunk_size or 0,
            kappa=args.kappa,
            sigma_low=args.kappa,
            sigma_high=args.kappa,
            winsor_limits=(args.winsor, args.winsor),
            parallel_io=True,
        )

        dest = os.path.join(args.out, "final.fits")
        shutil.move(final_path, dest)
        logger.info("Empilement final terminé: %s", dest)
        return 0

    # Auto-enable disk-backed alignment when processing extremely large
    # batches so memory usage stays bounded.  This mirrors the behaviour of
    # the GUI which suggests enabling the option for huge lists.
    if (
        args.batch_size == 1
        and not args.align_on_disk
        and len(rows) > 50
    ):
        logger.warning(
            "Large batch detected (>50 images) - enabling align_on_disk"
        )
        args.align_on_disk = True

    ordered_files = [r["path"] for r in rows]

    file_list = ordered_files
    out_path = os.path.join(args.out, "final.fits")
    hdr_ref = load_and_validate_fits(file_list[0])[1]
    # ``SeestarQueuedStacker`` expects ``input_dir`` to match the folder
    # containing ``stack_plan.csv`` when ``batch_size`` equals 1.  Using the
    # first image directory breaks this detection when files span multiple
    # subfolders, preventing the special single-batch mode from activating.
    # Point ``input_dir`` to the CSV location instead so the behaviour mirrors
    # the GUI's batch-size-1 handling ("mode 0").
    input_dir = os.path.dirname(os.path.abspath(args.csv))

    # ``tile`` is ignored by ``SeestarQueuedStacker`` but remains configurable
    # via ``SEESTAR_TILE_H`` to aid debugging memory usage.
    os.environ["SEESTAR_TILE_H"] = str(args.tile)

    bytes_limit = None
    if args.max_mem is not None:
        try:
            bytes_limit = int(float(args.max_mem) * 1024**3)
            os.environ["SEESTAR_MAX_MEM"] = str(bytes_limit)
        except (ValueError, TypeError):
            bytes_limit = None

    # When using Reproject+Coadd with single-image batches we must retain
    # each aligned FITS on disk so that an external astrometric solver can
    # update its WCS.  Force ``align_on_disk`` if the user hasn't enabled it.
    use_astrometric = (
        reproject_coadd_final
        and args.batch_size == 1
        and getattr(settings, "local_solver_preference", "none") != "none"
    )
    if use_astrometric and not args.align_on_disk:
        logger.info(
            "Reproject+Coadd avec résolution WCS requis --align-on-disk; activation automatique"
        )
        args.align_on_disk = True

    stacker = SeestarQueuedStacker(
        align_on_disk=args.align_on_disk,
        settings=settings,
        local_solver_preference=settings.local_solver_preference,
        astap_path=settings.astap_path,
        astap_data_dir=settings.astap_data_dir,
        astap_search_radius=settings.astap_search_radius,
        astap_downsample=settings.astap_downsample,
        astap_sensitivity=settings.astap_sensitivity,
    )
    global _GLOBAL_STACKER
    _GLOBAL_STACKER = stacker
    try:
        if bytes_limit is not None:
            try:
                stacker.max_hq_mem = bytes_limit
            except Exception:
                pass
        stacker.progress_callback = progress_cb

        if args.batch_size == 1 and stacker.astrometry_solver:
            _orig_parse_wcs = stacker.astrometry_solver._parse_wcs_file_content

            def _patched_parse_wcs_file_content(wcs_path, img_shape_hw, _orig=_orig_parse_wcs):
                _sanitize_astap_wcs(wcs_path)
                return _orig(wcs_path, img_shape_hw)

            stacker.astrometry_solver._parse_wcs_file_content = _patched_parse_wcs_file_content


        # ------------------------------------------------------------------
        # Pré-plate-solving des fichiers FITS sources (batch_size=1 uniquement)
        # ------------------------------------------------------------------
        solved_headers: list[fits.Header] = []
        if use_astrometric and args.batch_size == 1 and args.use_solver:
            for idx, src in enumerate(file_list):
                base = os.path.basename(src)
                try:
                    hdr = fits.getheader(src)
                except Exception:
                    hdr = fits.Header()

                has_wcs = False
                try:
                    WCS(hdr)
                    has_wcs = True
                except Exception:
                    has_wcs = False

                if has_wcs:
                    logger.info("WCS already present for %s", base)
                else:
                    if progress_cb:
                        progress_cb(f"WCS solving source: {base}", None)
                    try:
                        stacker._run_astap_and_update_header(src)
                        hdr = fits.getheader(src)
                    except Exception:
                        logger.exception("Astrometric solver failed for %s", src)
                        hdr = fits.Header()

                sanitize_header_for_wcs(hdr)
                try:
                    with fits.open(src, mode="update") as hdul:
                        hdul[0].header = hdr
                        hdul.flush()
                    logger.info("Header sanitized for %s", base)
                except Exception:
                    logger.exception("Failed to write sanitized header for %s", src)

                solved_headers.append(hdr)
        else:
            solved_headers = [fits.Header() for _ in file_list]

        _log_mem("before_start")  # DEBUG: baseline RAM
        ok = stacker.start_processing(
            input_dir=input_dir,
            output_dir=args.out,
            stacking_mode="winsorized-sigma",
            kappa=args.kappa,
            stack_kappa_low=args.kappa,
            stack_kappa_high=args.kappa,
            winsor_limits=(args.winsor, args.winsor),
            normalize_method=args.norm,
            weighting_method=args.weight,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            ordered_files=ordered_files,
            correct_hot_pixels=args.correct_hot_pixels,
            hot_pixel_threshold=args.hot_threshold,
            neighborhood_size=args.hot_neighborhood,
            use_weighting=args.use_weighting,
            snr_exp=args.snr_exp,
            stars_exp=args.stars_exp,
            min_w=args.min_weight,
            use_drizzle=False,
            reproject_between_batches=settings.reproject_between_batches,
            reproject_coadd_final=reproject_coadd_final,
            api_key=args.api_key,
            # When performing a final reproject/coadd with ``batch_size=1`` we
            # must keep the aligned files on disk for the last reprojection
            # step. Delay cleanup until the very end so external solvers
            # (ASTAP) can update the WCS headers of the aligned FITS files.
            perform_cleanup=(
                False
                if (
                    args.batch_size == 1
                    and final_combine in ("reproject", "reproject_coadd")
                )
                else args.cleanup_temp_files
            ),
        )
        _log_mem("after_start")  # DEBUG: after stacker launch
        if not ok:
            logger.error("start_processing failed")
            return 1

        start_time = time.monotonic()
        last_aligned = 0
        total_images = len(file_list)
        processed_temp_idx = 0

        def _process_new_aligned() -> None:
            """Attach solved WCS headers to aligned ``.npy`` files."""
            nonlocal processed_temp_idx

            def _has_essential_wcs(header: fits.Header) -> bool:
                required = {"CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"}
                has_scale = (
                    "CDELT1" in header
                    and "CDELT2" in header
                    or (
                        "CD1_1" in header
                        and "CD1_2" in header
                        and "CD2_1" in header
                        and "CD2_2" in header
                    )
                )
                return required.issubset(header.keys()) and has_scale

            if not use_astrometric:
                return
            new_paths = stacker.aligned_temp_paths[processed_temp_idx:]
            if not new_paths:
                return
            for fp in new_paths:
                base = os.path.basename(fp)
                hdr_path = fp.replace(".npy", ".hdr")
                hdr = fits.Header()
                try:
                    with open(hdr_path, "r", encoding="utf-8") as hf:
                        hdr = fits.Header.fromstring(hf.read(), sep="\n")
                except Exception as e_hdr:
                    logger.error("Header WCS introuvable pour %s: %s", base, e_hdr)
                    continue
                sanitize_header_for_wcs(hdr)
                if not _has_essential_wcs(hdr):
                    logger.error("Header WCS invalide pour %s", base)
                    continue
                try:
                    wcs_json = fp.replace(".npy", ".wcs.json")
                    with open(wcs_json, "w", encoding="utf-8") as jf:
                        json.dump({k: str(v) for k, v in hdr.items()}, jf, indent=2)
                    if progress_cb:
                        progress_cb(f"WCS stored: {base}", None)
                except Exception:
                    if progress_cb:
                        progress_cb(f"WCS store failure: {base}", None)
                if args.batch_size == 1:
                    try:
                        data = np.load(fp)
                        logger.debug(
                            "DEBUG BS: Chargé %s - shape=%s dtype=%s",
                            os.path.basename(fp),
                            data.shape,
                            data.dtype,
                        )
                        fits_path = fp.replace(".npy", ".fits")

                        expected_h = hdr.get("NAXIS2")
                        expected_w = hdr.get("NAXIS1")
                        if (
                            data.ndim != 3
                            or data.shape[2] != 3
                            or (expected_h and expected_w and data.shape != (expected_h, expected_w, 3))
                        ):
                            logger.error(
                                "Shape mismatch for %s: expected (%s,%s,3) got %s",
                                base,
                                expected_h,
                                expected_w,
                                data.shape,
                            )
                            continue

                        save_fits_image(data, fits_path, header=hdr, overwrite=True)
                        logger.info(
                            "FITS aligné exporté: %s", os.path.basename(fits_path)
                        )
                        # Validate header after write
                        try:
                            check_hdr = fits.getheader(fits_path)
                            if not _has_essential_wcs(check_hdr):
                                raise ValueError("missing essential WCS keys")
                            stacker.intermediate_classic_batch_files.append((fits_path, []))
                            logger.info(
                                "FITS aligné conservé pour reprojection finale: %s",
                                os.path.basename(fits_path),
                            )
                            if progress_cb:
                                progress_cb(
                                    f"Intermediary FITS created: {os.path.basename(fits_path)}",
                                    None,
                                )
                        except Exception as e_chk:
                            logger.error(
                                "Validation WCS échouée pour %s: %s", base, e_chk
                            )
                            try:
                                os.remove(fits_path)
                            except Exception:
                                pass
                    except Exception:
                        if progress_cb:
                            progress_cb(f"FITS export failure: {base}", None)
            processed_temp_idx += len(new_paths)

        while stacker.is_running():
            time.sleep(1)

            _process_new_aligned()

            current_aligned = getattr(stacker, "aligned_counter", 0)
            if progress_cb and current_aligned != last_aligned:
                last_aligned = current_aligned
                pct = current_aligned / max(total_images, 1) * 100
                elapsed = time.monotonic() - start_time
                if current_aligned > 0:
                    tpi = elapsed / current_aligned
                    remaining = max(total_images - current_aligned, 0)
                    eta_sec = remaining * tpi
                    h, r = divmod(int(eta_sec), 3600)
                    m, s = divmod(r, 60)
                    eta = f"{h:02}:{m:02}:{s:02}"
                else:
                    eta = "Calculating..."
                progress_cb(f"Aligned: {current_aligned}", pct)
                progress_cb(f"ETA_UPDATE:{eta}", None)

            getattr(stacker, "_indices_cache", {}).clear()
            gc.collect()
            _log_mem("loop")  # DEBUG: track RAM during run

        _process_new_aligned()  # catch any remaining files

        final_path = getattr(stacker, "final_stacked_path", None)
        final_reproject_success = False

        # Lorsque ``batch_size`` vaut 1 et que l'utilisateur a demandé une
        # reprojection finale, reprojeter toutes les images alignées sur la
        # grille de l'image de référence.
        if (
            args.batch_size == 1
            and final_combine in ("reproject", "reproject_coadd")
        ):
            aligned = [
                p[0] for p in getattr(stacker, "intermediate_classic_batch_files", [])
            ]
            aligned_existing = [f for f in aligned if os.path.isfile(f)]
            if not aligned_existing:
                raise RuntimeError(
                    "Reproject requested but no aligned FITS were produced."
                )
            reference_path = (
                getattr(settings, "reference_image_path", "") or aligned_existing[0]
            )
            out_fp = os.path.join(args.out, "final.fits")
            logger.info(
                "Lancement de la reprojection globale sur %d fichiers",
                len(aligned_existing),
            )
            logger.debug("First aligned FITS: %s", aligned_existing[0])
            logger.debug("Last aligned FITS: %s", aligned_existing[-1])
            t0 = time.monotonic()
            success = _finalize_reproject_and_coadd(
                aligned_existing, reference_path, out_fp
            )
            duration = time.monotonic() - t0
            logger.info("Reprojection globale terminée en %.2f s", duration)
            if not success:
                raise RuntimeError("Reproject and coadd failed.")
            logger.debug(
                "DEBUG: Reproject and coadd applied in boring stack (batch_size=1)"
            )
            final_path = out_fp
            final_reproject_success = True
        if final_path and os.path.isfile(final_path):
            dest = os.path.join(args.out, "final.fits")
            if os.path.abspath(final_path) != os.path.abspath(dest):
                shutil.copy2(final_path, dest)
            logger.info("Final FITS copied to %s", dest)
        preview = os.path.splitext(final_path)[0] + ".png" if final_path else None
        if preview and os.path.isfile(preview):
            shutil.copy2(preview, os.path.join(args.out, "preview.png"))

        _log_mem("after_copy")  # DEBUG: after writing results

        # ------------------------------------------------------------------
        # Cleanup of aligned temporary files after successful reprojection
        # ------------------------------------------------------------------
        if final_reproject_success and args.cleanup_temp_files and args.batch_size == 1:
            for npy_path in getattr(stacker, "aligned_temp_paths", []):
                for ext in (".npy", ".hdr", ".wcs.json", ".fits"):
                    tmp = npy_path.replace(".npy", ext)
                    if os.path.isfile(tmp):
                        try:
                            os.remove(tmp)
                            logger.info(
                                "Suppression post-reprojection du fichier aligné %s", tmp
                            )
                        except Exception:
                            logger.debug("Échec de suppression pour %s", tmp, exc_info=True)

        return 0
    finally:
        # Restore user's cleanup preference for stacker internals
        try:
            stacker.perform_cleanup = args.cleanup_temp_files
        except Exception:
            pass
        _cleanup_stacker(stacker)
        _log_mem("after_cleanup")  # DEBUG: final RAM state
        _GLOBAL_STACKER = None


def main() -> int:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Register signal handlers so that external termination requests
    # stop the stacker cleanly.
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            pass
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.out, "boring_stack.log"),
                mode="w",
                encoding="utf-8",
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )

    def log_progress(message: str, progress: object | None = None) -> None:
        """Simple progress callback that tolerates non-numeric ``progress``."""
        if progress is None:
            logger.info(message)
        elif isinstance(progress, (int, float)):
            logger.info("[%d%%] %s", int(progress), message)
        else:
            logger.info("%s (%s)", message, progress)
        _log_mem(f"progress:{message}")

    if args.show_progress and args.batch_size == 1:
        import threading
        import tkinter as tk
        from tkinter import ttk
        from .progress import ProgressManager

        root = tk.Tk()
        root.title("Boring Stack Progress")

        pb = ttk.Progressbar(root, mode="determinate", maximum=100)
        pb.pack(fill="x", padx=10, pady=5)

        status = tk.Text(root, height=8, state=tk.DISABLED)
        status.pack(fill="both", expand=True, padx=10, pady=5)

        rem_var = tk.StringVar(value="--:--:--")
        el_var = tk.StringVar(value="00:00:00")
        ttk.Label(root, text="Remaining:").pack(anchor="w", padx=10)
        ttk.Label(root, textvariable=rem_var).pack(anchor="w", padx=10)
        ttk.Label(root, text="Elapsed:").pack(anchor="w", padx=10)
        ttk.Label(root, textvariable=el_var).pack(anchor="w", padx=10)

        pm = ProgressManager(pb, status, rem_var, el_var)
        pm.reset()
        pm.start_timer()

        exit_code = 0

        def cb(msg, prog=None):
            pm.update_progress(msg, prog)

        def worker():
            nonlocal exit_code
            exit_code = _run_stack(args, cb)
            root.after(0, root.quit)

        threading.Thread(target=worker, daemon=True).start()
        root.mainloop()
        return exit_code

    if args.show_progress and args.batch_size != 1:
        logger.warning("--show-progress is only supported when batch_size=1")

    return _run_stack(args, log_progress)


if __name__ == "__main__":
    sys.exit(main())
