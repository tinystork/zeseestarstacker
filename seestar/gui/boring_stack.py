import argparse
import csv
import os
import sys
import gc
import ctypes
import logging

import numpy as np
from astropy.io import fits
import cv2
import imageio
from numpy.lib.format import open_memmap
from astropy.wcs import WCS



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



logger = logging.getLogger(__name__)


try:
    if "--out" in sys.argv:
        _init_logger(sys.argv[sys.argv.index("--out") + 1])
except Exception:
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(message)s")

# Allow running as a standalone script
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


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
    if arr.ndim == 3:
        # Typical case: channel-first (C, H, W)
        if arr.shape[0] <= 4:
            return arr.transpose(1, 2, 0)

        # Less common: (W, H, C) with header confirming orientation
        if (hdr is not None and arr.shape[2] <= 4 and hdr.get("NAXIS1") == arr.shape[0] and hdr.get("NAXIS2") == arr.shape[1]):
            return arr.transpose(1, 0, 2)

    return arr


def parse_args():
    p = argparse.ArgumentParser(description="Disk-based stacking script")
    p.add_argument("--csv", required=True, help="CSV with file list")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--tile", type=int, default=512, help="Tile height")
    p.add_argument("--kappa", type=float, default=3.0, help="Kappa value")
    p.add_argument("--winsor", type=float, default=0.05, help="Winsor limit")
    p.add_argument("--api-key", default=None, help="Astrometry.net API key")
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
    """Return list of paths only (legacy helper)."""
    return [r["path"] for r in read_rows(csv_path)]


def get_image_shape(path):
    """Return (height, width, channels) for an image file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".fit", ".fits"}:
        with fits.open(path, memmap=False) as hd:

            data = to_hwc(hd[0].data, hd[0].header)

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
        # Use ``memmap=True`` so only the requested slice is loaded in
        # memory.  ``memmap=False`` would read the entire image and quickly
        # exhaust RAM when stacking thousands of files.
        with fits.open(path, memmap=True) as hd:
            data = hd[0].data
            hdr = hd[0].header
        if data.ndim == 2 and hdr.get("BAYERPAT"):
            data = cv2.cvtColor(
                data.astype(np.uint16), cv2.COLOR_BayerRG2RGB_EA
            )
        else:
            data = to_hwc(data)
        data = data.astype(np.float32, copy=False)
    else:
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).astype(np.float32)

    if use_solver and wcs is not None and wcs_ref is not None:
        data = warp_image(data, wcs, wcs_ref, shape_ref)

    return data[y0:y1]


def winsorize(tile, kappa, limit):
    med = np.median(tile, axis=0, keepdims=True)
    mad = np.median(np.abs(tile - med), axis=0, keepdims=True) * 1.4826
    lo = med - kappa * mad
    hi = med + kappa * mad
    np.clip(tile, lo, hi, out=tile)
    return tile


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
):
    rows = read_rows(csv_path)
    if not rows:
        raise RuntimeError("CSV is empty")

    logger.info("Début du traitement")

    stream_stack._next_pct = 0.0

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
                print(
                    f"Solved {i}/{len(rows)}: {os.path.basename(path)} via {method}"
                )
            wcs_cache[path] = wcs

        if wcs_ref is not None:
            print("ALIGN OK")

        if wcs_ref is None:
            logger.warning("Reference WCS not resolved; stacking without alignment")
    else:
        for row in rows:
            wcs_cache[row["path"]] = None


    cum_sum = open_memmap(out_sum, "w+", dtype=np.float32, shape=(H, W, C))
    cum_sum[:] = 0
    cum_wht = open_memmap(out_wht, "w+", dtype=np.float32, shape=(H, W))
    cum_wht[:] = 1
    logger.debug(
        "allocated accumulators: cum_sum %s, cum_wht %s", cum_sum.shape, cum_wht.shape
    )

    tile_h = int(tile)
    for y0 in range(0, H, tile_h):
        y1 = min(y0 + tile_h, H)
        rows_h = y1 - y0
        tile_stack = [
            open_aligned_slice(
                r["path"],
                y0,
                y1,
                wcs_cache[r["path"]],
                wcs_ref,
                shape_ref,
                use_solver=use_solver,
            )
            for r in rows
        ]
        tile_wht_list = [
            np.full((rows_h, W), float(r.get("weight") or 1.0), np.float32)
            for r in rows
        ]
        tile = np.stack(tile_stack, axis=0)
        wht = np.stack(tile_wht_list, axis=0)
        winsorize(tile, kappa, winsor)
        cum_sum[y0:y1] += np.sum(tile * wht[..., None], axis=0)
        cum_wht[y0:y1] += np.sum(wht, axis=0)
        del tile_stack, tile_wht_list, tile, wht
        gc.collect()
        flush_mmap(cum_sum)
        flush_mmap(cum_wht)
        if y0 == 0:
            logger.debug(
                "stacked first tile -> cum_sum slice %s", cum_sum[y0:y1].shape
            )
        progress = 100.0 * y1 / H
        if progress >= stream_stack._next_pct or y1 == H:
            print(f"{progress:.1f}%", flush=True)
            stream_stack._next_pct = min(stream_stack._next_pct + 10.0, 100.0)

    return cum_sum, cum_wht


def main():
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)

    sum_path = os.path.join(args.out, "cum_sum.npy")
    wht_path = os.path.join(args.out, "cum_wht.npy")
    cum_sum, cum_wht = stream_stack(
        args.csv,
        sum_path,
        wht_path,
        tile=args.tile,
        kappa=args.kappa,
        winsor=args.winsor,
        api_key=args.api_key,
        use_solver=args.use_solver,
    )

    final = cum_sum / np.maximum(cum_wht[..., None], 1e-6)
    logger.debug("final image shape %s", final.shape)

    if final.ndim == 3 and final.shape[2] == 3:
        fits_data = final.transpose(2, 1, 0)
    else:
        fits_data = final.squeeze()

    fits.writeto(
        os.path.join(args.out, "final.fits"),
        fits_data.astype(np.float32),
        overwrite=True,
    )
    imageio.imwrite(
        os.path.join(args.out, "preview.png"),
        np.clip(final, 0, 1) ** 0.5,
    )
    flush_mmap(cum_sum)
    flush_mmap(cum_wht)
    del cum_sum
    del cum_wht
    gc.collect()
    try:
        os.remove(sum_path)
    except Exception as e:
        print(f"WARNING: could not remove {sum_path}: {e}", file=sys.stderr)
    try:
        os.remove(wht_path)
    except Exception as e:
        print(f"WARNING: could not remove {wht_path}: {e}", file=sys.stderr)
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
        stream_stack(
            csv_path,
            os.path.join(tmp, "sum.npy"),
            os.path.join(tmp, "wht.npy"),
        )
        arr = np.load(os.path.join(tmp, "sum.npy"), mmap_mode="r")
        assert arr.shape == (4, 5, 3), arr.shape
        shutil.rmtree(tmp)
        sys.exit(0)

    try:
        sys.exit(main())
    except Exception:
        logging.exception("Fatal error in boring_stack")
        sys.exit(1)
