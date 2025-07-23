import argparse
import csv
import os
import sys
import time
import logging
import shutil
import tempfile

from numpy.lib.format import open_memmap as _orig_open_memmap


def _safe_open_memmap(filename, *args, **kwargs):
    """Fallback to a temp directory if the original path fails."""
    try:
        return _orig_open_memmap(filename, *args, **kwargs)
    except OSError:
        tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(str(filename)))
        return _orig_open_memmap(tmp_path, *args, **kwargs)

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
    p.set_defaults(use_solver=True)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main processing logic
# -----------------------------------------------------------------------------

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

    rows = read_rows(args.csv)
    if not rows:
        logger.error("CSV is empty")
        return 1

    ordered_files = [r["path"] for r in rows]
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

    stacker = SeestarQueuedStacker()
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
        batch_size=1,
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
    if not ok:
        logger.error("start_processing failed")
        return 1

    while stacker.is_running():
        time.sleep(1)

    final_path = getattr(stacker, "final_stacked_path", None)
    if final_path and os.path.isfile(final_path):
        dest = os.path.join(args.out, "final.fits")
        shutil.copy2(final_path, dest)
        logger.info("Final FITS copied to %s", dest)
    preview = os.path.splitext(final_path)[0] + ".png" if final_path else None
    if preview and os.path.isfile(preview):
        shutil.copy2(preview, os.path.join(args.out, "preview.png"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
