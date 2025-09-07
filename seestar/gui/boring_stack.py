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

from logging.handlers import RotatingFileHandler


# Ensure the project root is on sys.path when executed directly
if __package__ in (None, ""):
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:  # Allow tests to stub core package without providing reproject_utils
    from seestar import reproject_utils
except Exception:  # pragma: no cover - missing during certain tests
    reproject_utils = None  # type: ignore

try:  # Optional during tests that stub core modules
    from seestar.core.reprojection_utils import (
        collect_headers,
        compute_final_output_grid,
    )
except Exception:  # pragma: no cover - allow stubbing in tests
    collect_headers = compute_final_output_grid = None  # type: ignore


def _setup_logging(log_dir: str, log_name: str, level: int) -> str:
    """Configure root logging for both CLI and GUI launches."""
    os.makedirs(log_dir, exist_ok=True)
    log_fp = os.path.join(log_dir, log_name)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = RotatingFileHandler(log_fp, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    root.setLevel(level)
    logging.getLogger(__name__).propagate = True
    return log_fp




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
        base = os.path.basename(str(filename))
        fd, tmp_path = tempfile.mkstemp(prefix=base + "_", suffix=".npy")
        os.close(fd)
        try:
            mm = _orig_open_memmap(tmp_path, *args, **kwargs)
        except OSError as e:
            raise OSError(
                f"Unable to create memmap at '{filename}' or fallback '{tmp_path}'"
            ) from e
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
# Import helpers from the core package.  Use an absolute import so that this
# script can be executed directly without Python considering it a package
# relative module.  When ``__package__`` is ``None`` the project root is added
# to ``sys.path`` above, allowing this import to succeed.
from seestar.core.image_processing import (
    load_and_validate_fits,
    save_fits_image,
)
from astropy.io import fits
from astropy.wcs import WCS
from seestar.alignment.astrometry_solver import AstrometrySolver
from seestar.utils.wcs_utils import _sanitize_continue_as_string, write_wcs_to_fits_inplace
import glob
from astropy.io.fits.verify import VerifyWarning
import warnings

try:
    from seestar.queuep.queue_manager import renormalize_fits as _qm_renorm
except Exception:  # pragma: no cover - best effort
    _qm_renorm = None

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# WCS validators
# -----------------------------------------------------------------------------


def _has_essential_wcs(h: fits.Header) -> bool:
    need = {"CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"}
    has_scale = (
        ("CDELT1" in h and "CDELT2" in h)
        or all(k in h for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"))
    )
    return need.issubset(h.keys()) and has_scale


def _wcs_is_valid_celestial(hdr: fits.Header) -> bool:
    try:

        hdr = reproject_utils.sanitize_header_for_wcs(hdr.copy())
        w = WCS(hdr, naxis=2)
        return reproject_utils.is_valid_celestial_wcs(w)

    except Exception:
        return False


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


def _has_wcs(hdr: fits.Header) -> bool:
    return (
        "CRVAL1" in hdr
        and "CRVAL2" in hdr
        and (
            ("CD1_1" in hdr and "CD2_2" in hdr)
            or "PC1_1" in hdr
            or "CDELT1" in hdr
        )
    )


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
    missing_count = 0
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
        if not os.path.isfile(cell):
            missing_count += 1
            logger.warning("Stack plan path not found: %s", cell)
            continue
        weight = ""
        if weight_idx is not None and len(row) > weight_idx:
            weight = row[weight_idx].strip()
        rows_out.append({"path": cell, "weight": weight})
    if missing_count:
        logger.warning("Skipped %d stack plan entries with missing files", missing_count)
    return rows_out


def read_paths(csv_path):
    return [r["path"] for r in read_rows(csv_path)]


# -----------------------------------------------------------------------------
# Reproject+Coadd helpers
# -----------------------------------------------------------------------------


def _load_wcs_header_only(fp: str) -> WCS:
    """Load a WCS from a FITS file without touching the data."""
    # ``ignore_missing_simple`` avoids ``VerifyError`` when the FITS file lacks
    # a ``SIMPLE`` card (observed with batch_size=1 on some ASTAP outputs).
    # The ``memmap=True`` flag keeps the disk-based workflow unchanged.
    with fits.open(fp, memmap=True, ignore_missing_simple=True) as hdul:
        hdr = hdul[0].header.copy()
    _sanitize_continue_as_string(hdr)
    return WCS(hdr, naxis=2, relax=True)


def _finalize_reproject_and_coadd(
    arg1,
    arg2,
    arg3=None,
    *,
    prefer_streaming_fallback: bool = True,
    tile_size: Optional[int] = None,
    output_wcs=None,
    shape_out=None,
) -> bool:
    """Finalize a reprojection + coadd operation.

    Two calling conventions are supported for backward compatibility:

    1. ``_finalize_reproject_and_coadd(aligned_dir, out_fp)``
       where ``aligned_dir`` contains ``aligned_*.fits`` files.
    2. ``_finalize_reproject_and_coadd(paths, reference_path, out_fp)``
       where ``paths`` is an iterable of FITS files to reproject and coadd.
    """

    if arg3 is None:
        aligned_dir, out_fp = arg1, arg2
        files = sorted(glob.glob(os.path.join(aligned_dir, "aligned_*.fits")))
        if not files:
            logger.error("No valid WCS candidates -> abort to avoid empty mosaic.")
            return False
        paths = files
        ref_fp = files[0]
        n_inputs = len(files)
    else:
        paths, ref_fp, out_fp = arg1, arg2, arg3
        paths = list(paths)
        if ref_fp is not None:
            paths = [ref_fp] + paths
        elif paths:
            ref_fp = paths[0]
        n_inputs = len(paths)

    if paths:
        to_log = [paths[0]]
        if len(paths) > 1:
            to_log.append(paths[-1])
        for fp in to_log:
            try:
                # Some aligned FITS produced by ASTAP/processing have
                # BZERO/BSCALE/BLANK in the header which prevents memmap reads
                # in Astropy. For logging we only need shape + a few WCS
                # keywords, so open without memmap to avoid the ValueError.
                with fits.open(fp, memmap=False) as hdul:
                    data_shape = getattr(hdul[0].data, "shape", None)
                    hdr = hdul[0].header
                nax1 = hdr.get("NAXIS1")
                nax2 = hdr.get("NAXIS2")
                pc = [[hdr.get("PC1_1"), hdr.get("PC1_2")], [hdr.get("PC2_1"), hdr.get("PC2_2")]]
                if not any(v is not None for row in pc for v in row):
                    pc = [[hdr.get("CD1_1"), hdr.get("CD1_2")], [hdr.get("CD2_1"), hdr.get("CD2_2")]]
                logger.info(
                    "[BS=1][CHECK] %s data.shape=%s NAXIS1=%s NAXIS2=%s matrix=%s",
                    os.path.basename(fp), data_shape, nax1, nax2, pc,
                )
            except Exception:
                logger.exception("Failed to log aligned file %s", fp)

    if output_wcs is not None and shape_out is not None:
        ref_wcs = output_wcs if isinstance(output_wcs, WCS) else WCS(output_wcs)
        h, w = int(shape_out[0]), int(shape_out[1])
        logger.info("[BS=1][COADD] Using fixed grid shape_out=%dx%d", h, w)
        result = reproject_utils.reproject_and_coadd_from_paths(
            paths,
            output_projection=ref_wcs,
            shape_out=shape_out,
            prefer_streaming_fallback=prefer_streaming_fallback,
            tile_size=tile_size,
            match_background=False,
        )
        hdr_out = ref_wcs.to_header(relax=True)
    else:
        # Use the reference file's WCS and dimensions as the target projection so
        # the reprojection grid exactly matches the stack reference.  This avoids
        # subtle sub-pixel offsets that could occur when letting
        # ``reproject_and_coadd_from_paths`` auto-derive an output grid, which was
        # causing ghosting artefacts for ``batch_size=1`` stacks.
        try:
            ref_hdr = fits.getheader(
                ref_fp, memmap=False, ignore_missing_simple=True
            )
            ref_hdr = reproject_utils.sanitize_header_for_wcs(ref_hdr)
            ref_wcs = WCS(ref_hdr, naxis=2)
            h = int(ref_hdr.get("NAXIS2", 0))
            w = int(ref_hdr.get("NAXIS1", 0))
            shape_out = (h, w) if h > 0 and w > 0 else None
            result = reproject_utils.reproject_and_coadd_from_paths(
                paths,
                output_projection=ref_wcs,
                shape_out=shape_out,
                prefer_streaming_fallback=prefer_streaming_fallback,
                tile_size=tile_size,
                match_background=False,
            )
            hdr_out = ref_wcs.to_header(relax=True)
        except Exception as e:
            logger.warning(
                "Reference WCS invalid: %s -> fallback to auto grid", e
            )
            result = reproject_utils.reproject_and_coadd_from_paths(
                paths,
                prefer_streaming_fallback=prefer_streaming_fallback,
                tile_size=tile_size,
                match_background=False,
            )
            hdr_out = result.wcs.to_header(relax=True)

    wht = np.asarray(getattr(result, "weight", []))
    if (
        output_wcs is None
        and (wht.size == 0 or not np.isfinite(wht).any() or float(np.nanmax(wht)) <= 0)
    ):
        logger.warning(
            "Reference-grid reprojection produced empty mosaic -> fallback to auto grid"
        )
        result = reproject_utils.reproject_and_coadd_from_paths(
            paths,
            prefer_streaming_fallback=prefer_streaming_fallback,
            tile_size=tile_size,
            match_background=False,
        )
        hdr_out = result.wcs.to_header(relax=True)

    for k in list(hdr_out.keys()):
        if k == "SIMPLE" or k.startswith("NAXIS") or k in (
            "BITPIX",
            "EXTEND",
            "PCOUNT",
            "GCOUNT",
            "XTENSION",
        ):
            del hdr_out[k]

    if (
        result.image.ndim == 3
        and result.image.shape[-1] in (3, 4)
        and result.image.shape[0] not in (1, 3, 4)
    ):
        data_out = np.moveaxis(result.image, -1, 0)
    else:
        data_out = result.image

    if np.issubdtype(data_out.dtype, np.floating):
        for k in ("BSCALE", "BZERO"):
            if k in hdr_out:
                del hdr_out[k]

    if isinstance(data_out, np.ndarray) and data_out.dtype != np.float32:
        data_out = data_out.astype(np.float32, copy=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=VerifyWarning)
        fits.PrimaryHDU(data=data_out, header=hdr_out).writeto(
            out_fp, overwrite=True
        )

    # ---- Renormalisation post-écriture : BS=1 uniquement ----
    try:
        n_inputs = int(n_inputs) if "n_inputs" in locals() else None
        if n_inputs is None or n_inputs <= 0:
            n_inputs = 1

        if _qm_renorm is not None:
            _qm_renorm(out_fp, method="n_images", n_images=n_inputs)
        else:
            with fits.open(out_fp, mode="update") as hdul:
                data = hdul[0].data
                hdr = hdul[0].header

                scale = float(n_inputs)
                if np.nanmax(np.abs(data)) < 1e-12:
                    scale = 1.0

                data = (data * scale).astype(np.float32, copy=False)
                hdul[0].data = data

                hdr["ZNORM"] = ("N_IMAGES", "Output normalized by number of inputs")
                hdr["ZNSCALE"] = (float(scale), "Normalization factor applied")
                hdr["NINPUTS"] = (int(n_inputs), "Number of images used in coadd")
                hdr.add_history("BS=1 normalization applied (method=N_IMAGES).")

                hdul.flush()
        logger.info(f"[BS=1] Normalized output by N_IMAGES: factor={n_inputs}")
    except Exception as _e:
        logger.warning(f"[BS=1] Normalization step failed: {type(_e).__name__}: {_e}")

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


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Batch-1 stacking using QueueManager")
    # Weight options mirror the GUI: snr, stars, noise variance and noise+FWHM
    p.add_argument("--csv", required=True, help="CSV with file list")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--log-dir", default=None, help="Directory for log file")
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
    p.add_argument("--tile-size", type=int, default=1024, help="Tile size for streaming reprojection")
    p.add_argument("--dtype-out", default="float32", choices=["float32", "float64"], help="Output dtype for streaming reprojection")
    # Whether to save the final FITS as float32 (matches GUI option)
    p.add_argument(
        "--save-as-float32",
        dest="save_as_float32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save final FITS as float32 (or uint16 if disabled)",
    )
    p.add_argument("--memmap-dir", default=None, help="Directory for temporary memmap files")
    p.add_argument("--keep-intermediates", action="store_true", help="Keep temporary memmap files")
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
        logger.error("No valid file paths found in CSV")
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

    solver_settings = {
        "local_solver_preference": settings.local_solver_preference,
        "api_key": args.api_key or getattr(settings, "astrometry_api_key", ""),
        "astap_path": settings.astap_path,
        "astap_data_dir": settings.astap_data_dir,
        "astap_search_radius": settings.astap_search_radius,
        "astap_downsample": settings.astap_downsample,
        "astap_sensitivity": settings.astap_sensitivity,
        "local_ansvr_path": getattr(settings, "local_ansvr_path", ""),
    }

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
    stacker.api_key = solver_settings["api_key"]
    solver = AstrometrySolver(progress_callback=progress_cb) if args.batch_size == 1 else None
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

                has_wcs = _wcs_is_valid_celestial(hdr)

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

                hdr = reproject_utils.sanitize_header_for_wcs(hdr)
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
            # Ensure final FITS dtype matches user's setting or CLI override
            save_as_float32=(
                args.save_as_float32
                if getattr(args, "save_as_float32", None) is not None
                else getattr(settings, "save_final_as_float32", False)
            ),
            **solver_settings,
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

            if not use_astrometric:
                return
            new_paths = stacker.aligned_temp_paths[processed_temp_idx:]
            if not new_paths:
                return
            for fp in new_paths:
                base = os.path.basename(fp)
                root, ext = os.path.splitext(fp)
                hdr_path = root + ".hdr"
                hdr = fits.Header()
                try:
                    with open(hdr_path, "r", encoding="utf-8") as hf:
                        hdr = fits.Header.fromstring(hf.read(), sep="\n")
                except Exception as e_hdr:
                    logger.error("Header WCS introuvable pour %s: %s", base, e_hdr)
                    continue
                reproject_utils.sanitize_header_for_wcs(hdr)
                if not _has_essential_wcs(hdr):
                    logger.error("Header WCS invalide pour %s", base)
                    continue
                try:
                    wcs_obj = WCS(hdr, naxis=2)
                except Exception as e_wcs:
                    logger.error("Header-WCS invalid for %s: %s", base, e_wcs)
                    continue
                try:
                    wcs_json = root + ".wcs.json"
                    with open(wcs_json, "w", encoding="utf-8") as jf:
                        json.dump({k: str(v) for k, v in hdr.items()}, jf, indent=2)
                    if progress_cb:
                        progress_cb(f"WCS stored: {base}", None)
                except Exception:
                    if progress_cb:
                        progress_cb(f"WCS store failure: {base}", None)
                if args.batch_size == 1:
                    try:
                        if ext.lower() == ".npy":
                            data = np.load(fp)
                        elif ext.lower() == ".fits":
                            data = fits.getdata(fp, memmap=False)
                            if data.ndim == 2:
                                data = np.dstack([data] * 3)
                            elif data.ndim == 3 and data.shape[0] in (3, 4):
                                data = np.moveaxis(data, 0, -1)
                        else:
                            logger.error("Unsupported temp file type: %s", fp)
                            continue
                        logger.debug(
                            "DEBUG BS: Chargé %s - shape=%s dtype=%s",
                            os.path.basename(fp),
                            data.shape,
                            data.dtype,
                        )
                        fits_path = root + ".fits" if ext.lower() == ".npy" else fp

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

        
                        if ext.lower() == ".npy":
                            save_fits_image(data, fits_path, header=hdr, overwrite=True)
                            logger.info(
                                "FITS aligné exporté: %s", os.path.basename(fits_path)
                            )
                        hdr_aligned = fits.getheader(fits_path, 0)
                        if _has_essential_wcs(hdr_aligned) and _wcs_is_valid_celestial(
                            hdr_aligned
                        ):
                            logger.info(
                                "[BS=1][WCS] Skip sidecar: aligned FITS already has a valid celestial WCS → %s",
                                os.path.basename(fits_path),
                            )
                        else:
                            try:
                                write_wcs_to_fits_inplace(fits_path, wcs_obj)
                                logger.info(
                                    "[BS=1][WCS] Inject sidecar WCS into aligned FITS → %s",
                                    os.path.basename(fits_path),
                                )
                            except Exception as e_w:
                                logger.warning(
                                    "Persist WCS failed for %s: %s", fits_path, e_w
                                )
                        # Validate header after skip/injection
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

            aligned_dir = getattr(stacker, "aligned_temp_dir", "")
            files = sorted(glob.glob(os.path.join(aligned_dir, "aligned_*.fits")))
            if not files:
                raise RuntimeError(
                    "Reproject requested but no aligned FITS were produced."
                )
            out_fp = os.path.join(args.out, "final.fits")

            # Use the WCS of the stacking reference so the final grid matches
            # the reference orientation (mirrors batch_size=0 behaviour).
            ref_wcs = getattr(stacker, "reference_wcs_object", None)
            ref_shape = getattr(stacker, "reference_shape", None)

            # In ``batch_size=1`` mode we want the final reprojection grid to
            # mirror the live-stacking behaviour (``batch_size=0``) which uses
            # the reference frame's original dimensions.  The queue manager may
            # carry a global mosaic shape (e.g. 3840x2160) but here we ignore it
            # and derive the target grid from the first aligned file instead.
            if args.batch_size == 1:
                ref_wcs = None
                ref_shape = None
            elif (ref_wcs is None or ref_shape is None) and getattr(
                stacker, "output_folder", None
            ):
                ref_fp = os.path.join(
                    stacker.output_folder, "temp_processing", "reference_image.fit"
                )
                if os.path.isfile(ref_fp):
                    try:
                        hdr = fits.getheader(
                            ref_fp, memmap=False, ignore_missing_simple=True
                        )
                        hdr = reproject_utils.sanitize_header_for_wcs(hdr)
                        ref_wcs = WCS(hdr, naxis=2)
                        h = int(hdr.get("NAXIS2", 0))
                        w = int(hdr.get("NAXIS1", 0))
                        if h > 0 and w > 0:
                            ref_shape = (h, w)
                    except Exception:
                        pass

            t0 = time.monotonic()
            success = _finalize_reproject_and_coadd(
                files,
                None,
                out_fp,
                prefer_streaming_fallback=True,
                tile_size=getattr(args, "tile", None),
                output_wcs=ref_wcs,
                shape_out=ref_shape,
            )
            duration = time.monotonic() - t0
            logger.info("Final reprojection+coadd done in %.2f s", duration)
            if not success:
                raise RuntimeError("Reproject and coadd failed.")
            with fits.open(out_fp, memmap=False) as hdul:
                logger.info(
                    "Final written: %s  (H, W)=%s", out_fp, hdul[0].data.shape
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
            for img_path in getattr(stacker, "aligned_temp_paths", []):
                base = os.path.splitext(img_path)[0]
                for ext in (".fits", ".hdr", ".wcs.json", ".npy"):
                    tmp = base + ext
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
    log_dir = args.log_dir or os.path.join(args.out, "logs")
    _setup_logging(log_dir, "seestar.log", logging.DEBUG)

    # Register signal handlers so that external termination requests
    # stop the stacker cleanly.
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            pass

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
