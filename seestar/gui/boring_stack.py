import argparse
import csv
import os
import sys
import gc
import ctypes
import traceback

import numpy as np
from astropy.io import fits
import cv2
from numpy.lib.format import open_memmap


def to_hwc(arr: np.ndarray, hdr: fits.Header | None = None) -> np.ndarray:
    """Return ``arr`` in ``(H, W, C)`` order if necessary."""
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
    return p.parse_args()


def read_paths(csv_path):
    """Read file paths from ``csv_path`` respecting optional headers.

    The CSV may contain a simple list of paths or a full ``stack_plan.csv``
    with additional columns.  In the latter case the ``file_path`` column is
    used.  A header row like ``order,file`` or ``index,file`` is also
    supported.  Relative paths are resolved against the CSV location.
    """

    files: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return files

    header = [c.strip().lower() for c in rows[0]]
    file_idx = None
    data_rows = rows

    if "file_path" in header:
        file_idx = header.index("file_path")
        data_rows = rows[1:]
    else:
        has_header = any(
            h in {"order", "file", "filename", "path", "index"} for h in header
        )
        if has_header:
            data_rows = rows[1:]

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

        files.append(cell)

    return files


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
    return h, w, c


def open_slice(path, y0, y1):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".fit", ".fits"}:
        # Using memmap=True fails when FITS files contain scaling keywords such
        # as BZERO/BSCALE/BLANK. These are fairly common and cause astropy to
        # raise a ValueError because the data cannot be memory mapped. Reading
        # the data without memory mapping avoids this issue while keeping the
        # rest of the logic unchanged.
        with fits.open(path, memmap=False) as hd:

            data = to_hwc(hd[0].data, hd[0].header)[y0:y1]

            arr = data.astype(np.float32, copy=False)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read {path}")
        arr = img[y0:y1].astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


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


def stream_stack(csv_path, out_sum, out_wht, *, tile=512, kappa=3.0, winsor=0.05):
    files = read_paths(csv_path)
    if not files:
        raise RuntimeError("CSV is empty")

    first = files[0]
    H, W, C = get_image_shape(first)

    cum_sum = open_memmap(out_sum, "w+", dtype=np.float32, shape=(H, W, C))
    cum_sum[:] = 0
    cum_wht = open_memmap(out_wht, "w+", dtype=np.float32, shape=(H, W))
    cum_wht[:] = 1

    tile_h = int(tile)
    for y0 in range(0, H, tile_h):
        y1 = min(y0 + tile_h, H)
        tile_slices = [open_slice(fp, y0, y1) for fp in files]
        tile = np.stack(tile_slices, axis=0)
        winsorize(tile, kappa, winsor)
        cum_sum[y0:y1] += tile.sum(axis=0)
        cum_wht[y0:y1] += tile.shape[0]
        del tile_slices
        del tile
        gc.collect()
        flush_mmap(cum_sum)
        flush_mmap(cum_wht)
        progress = 100.0 * y1 / H
        print(f"{progress:.1f}%", flush=True)

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
    )

    final = cum_sum / np.maximum(cum_wht[..., None], 1e-6)
    fits.writeto(
        os.path.join(args.out, "final.fits"),
        final.astype(np.float32),
        overwrite=True,
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

        tmp = tempfile.mkdtemp()
        fits.writeto(
            os.path.join(tmp, "c_hw.fits"),
            (np.arange(60, dtype=np.uint16).reshape(3, 4, 5)),
            overwrite=True,
        )
        csv_path = os.path.join(tmp, "plan.csv")
        with open(csv_path, "w") as f:
            f.write("file_path\n" + tmp + "/c_hw.fits\n")
        stream_stack(csv_path, os.path.join(tmp, "sum.npy"), os.path.join(tmp, "wht.npy"))
        arr = np.load(os.path.join(tmp, "sum.npy"), mmap_mode="r")
        assert arr.shape == (4, 5, 3), arr.shape
        shutil.rmtree(tmp)
        sys.exit(0)

    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc(limit=3)
        sys.exit(1)
