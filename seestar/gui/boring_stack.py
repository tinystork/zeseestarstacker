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
import numpy as np

_PROC = psutil.Process(os.getpid())  # Track process RAM for DEBUG logs

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

import numpy.lib.format as _np_format
_np_format.open_memmap = _safe_open_memmap

from seestar.queuep.queue_manager import SeestarQueuedStacker
# Import helpers from the core package.  Use an absolute import so that this
# script can be executed directly without Python considering it a package
# relative module.  When ``__package__`` is ``None`` the project root is added
# to ``sys.path`` above, allowing this import to succeed.
from seestar.core.image_processing import load_and_validate_fits, save_fits_image

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
    p.add_argument("--max-mem", type=float, default=None, help="(unused)")
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

    stacker = SeestarQueuedStacker(align_on_disk=args.align_on_disk)
    try:
        stacker.progress_callback = progress_cb
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
            reproject_between_batches=False,
            api_key=args.api_key,
            perform_cleanup=args.cleanup_temp_files,
        )
        _log_mem("after_start")  # DEBUG: after stacker launch
        if not ok:
            logger.error("start_processing failed")
            return 1

        while stacker.is_running():
            time.sleep(1)
            getattr(stacker, "_indices_cache", {}).clear()
            gc.collect()
            _log_mem("loop")  # DEBUG: track RAM during run

        final_path = getattr(stacker, "final_stacked_path", None)
        if final_path and os.path.isfile(final_path):
            dest = os.path.join(args.out, "final.fits")
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


def main() -> int:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
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
