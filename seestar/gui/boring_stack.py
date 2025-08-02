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
# Import helpers from the core package.  Use an absolute import so that this
# script can be executed directly without Python considering it a package
# relative module.  When ``__package__`` is ``None`` the project root is added
# to ``sys.path`` above, allowing this import to succeed.
from seestar.core.image_processing import load_and_validate_fits, save_fits_image
from seestar.enhancement.reproject_utils import reproject_and_coadd
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


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

def _finalize_reproject_and_coadd(aligned_files, output_path):
    """Combine ``aligned_files`` using :func:`reproject_and_coadd`."""

    input_pairs = []
    for fp in aligned_files:
        try:
            with fits.open(fp, memmap=False) as hdul:
                data = hdul[0].data.astype(np.float32)
                hdr = hdul[0].header
        except Exception:
            continue
        if data.ndim == 3 and data.shape[0] in (1, 3) and data.shape[0] != data.shape[-1]:
            data = np.moveaxis(data, 0, -1)
        wcs = WCS(hdr, naxis=2)
        if wcs.pixel_shape is None:
            h, w = data.shape[:2]
            wcs.pixel_shape = (w, h)
        input_pairs.append((data, wcs))

    if not input_pairs:
        return False

    output_projection = input_pairs[0][1]
    shape_out = input_pairs[0][0].shape[:2]

    stacked, _ = reproject_and_coadd(
        input_pairs,
        output_projection=output_projection,
        shape_out=shape_out,
        combine_function="mean",
        match_background=True,
    )

    fits.PrimaryHDU(
        data=np.moveaxis(stacked, -1, 0),
        header=output_projection.to_header(relax=True),
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
    p.set_defaults(use_solver=True)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main processing logic
# -----------------------------------------------------------------------------

def _run_stack(args, progress_cb) -> int:
    """Execute the stacking process using ``progress_cb`` for updates."""
    rows = read_rows(args.csv)
    if not rows:
        logger.error("CSV is empty")
        return 1

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

    is_plan = os.path.basename(args.csv).lower() == "stack_plan.csv"
    if (
        args.batch_size == 1
        and is_plan
        and args.norm in {"linear_fit", "sky_mean"}
    ):
        norm_dir = os.path.join(args.out, "normalized_inputs")
        os.makedirs(norm_dir, exist_ok=True)
        if args.norm == "linear_fit":
            coeffs = _calculate_global_linear_fit(ordered_files[0], ordered_files)
        else:
            target_sky = _calculate_global_sky_mean(ordered_files)
        normalized_rows = []
        for idx, (row, path) in enumerate(zip(rows, ordered_files)):
            data, hdr = load_and_validate_fits(path)
            if data is None:
                continue
            if args.norm == "linear_fit":
                slope, inter = coeffs[idx]
                data = _apply_linear_fit(data, slope, inter)
            else:
                data = _apply_sky_mean(data, target_sky)
            out_fp = os.path.join(norm_dir, f"{idx:04d}.fits")
            save_fits_image(data, out_fp, hdr)
            normalized_rows.append({"path": out_fp, "weight": row.get("weight", "")})
        rows = normalized_rows
        ordered_files = [r["path"] for r in rows]
        args.norm = "none"

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

    # Load user settings so final combine preferences are honoured
    settings = SettingsManager()
    try:
        settings.load_settings()
    except Exception:
        settings.reset_to_defaults()

    # When using Reproject+Coadd with single-image batches we must retain
    # each aligned FITS on disk so that an external astrometric solver can
    # update its WCS.  Force ``align_on_disk`` if the user hasn't enabled it.
    use_astrometric = (
        settings.reproject_coadd_final
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

        # ------------------------------------------------------------------
        # Pré-plate-solving des fichiers FITS sources (batch_size=1 uniquement)
        # ------------------------------------------------------------------
        solved_headers: list[fits.Header] = []
        if use_astrometric and args.batch_size == 1 and args.use_solver:
            for idx, src in enumerate(file_list):
                base = os.path.basename(src)
                if progress_cb:
                    progress_cb(f"WCS solving source: {base}", None)
                try:
                    stacker._run_astap_and_update_header(src)
                except Exception:
                    logger.exception("Astrometric solver failed for %s", src)
                try:
                    solved_headers.append(fits.getheader(src))
                except Exception:
                    solved_headers.append(fits.Header())
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
            reproject_coadd_final=settings.reproject_coadd_final,
            api_key=args.api_key,
            perform_cleanup=args.cleanup_temp_files,
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
                idx_str = os.path.splitext(base)[0].split("_")[-1]
                try:
                    idx = int(idx_str)
                except ValueError:
                    idx = -1
                hdr = solved_headers[idx] if 0 <= idx < len(solved_headers) else fits.Header()
                try:
                    wcs_json = fp.replace(".npy", ".wcs.json")
                    with open(wcs_json, "w", encoding="utf-8") as jf:
                        json.dump({k: str(v) for k, v in hdr.items()}, jf, indent=2)
                    if progress_cb:
                        progress_cb(f"WCS stored: {base}", None)
                except Exception:
                    if progress_cb:
                        progress_cb(f"WCS store failure: {base}", None)
                if settings.reproject_coadd_final:
                    try:
                        data = np.load(fp)
                        fits_path = fp.replace(".npy", ".fits")
                        fits.PrimaryHDU(data=data, header=hdr).writeto(
                            fits_path, overwrite=True
                        )
                        stacker.intermediate_classic_batch_files.append((fits_path, []))
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

        # Explicitly handle the "Reproject and coadd" final combine mode when
        # running in single-image batches. All intermediate aligned files are
        # combined using :func:`reproject_and_coadd` to produce the final image.
        if settings.reproject_coadd_final and args.batch_size == 1:
            aligned = [
                p[0] for p in getattr(stacker, "intermediate_classic_batch_files", [])
            ]
            logger.info(
                "Fichiers alignés initialement détectés : %s", aligned
            )
            aligned_existing = [f for f in aligned if os.path.isfile(f)]
            logger.info(
                "Fichiers alignés existants : %s", aligned_existing
            )
            if aligned_existing:
                out_fp = os.path.join(args.out, "final.fits")
                success = _finalize_reproject_and_coadd(aligned_existing, out_fp)
                if success:
                    final_path = out_fp
                    logger.info(
                        "Image finale créée avec succès via reproject_and_coadd."
                    )
                else:
                    logger.error(
                        "Échec de _finalize_reproject_and_coadd malgré l'existence des fichiers alignés."
                    )
            else:
                logger.error(
                    "Aucun fichier aligné valide trouvé pour la reprojection finale."
                )
        if final_path and os.path.isfile(final_path):
            dest = os.path.join(args.out, "final.fits")
            if os.path.abspath(final_path) != os.path.abspath(dest):
                shutil.copy2(final_path, dest)
            logger.info("Final FITS copied to %s", dest)
        preview = os.path.splitext(final_path)[0] + ".png" if final_path else None
        if preview and os.path.isfile(preview):
            shutil.copy2(preview, os.path.join(args.out, "preview.png"))

        _log_mem("after_copy")  # DEBUG: after writing results

        return 0
    finally:
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
