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


def parse_args():
    p = argparse.ArgumentParser(description="Disk-based stacking script")
    p.add_argument("--csv", required=True, help="CSV with file list")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--tile", type=int, default=512, help="Tile height")
    p.add_argument("--kappa", type=float, default=3.0, help="Kappa value")
    p.add_argument("--winsor", type=float, default=0.05, help="Winsor limit")
    return p.parse_args()


def read_paths(csv_path):
    files = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if row:
                files.append(row[0])
    return files


def open_slice(path, y0, y1):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".fit", ".fits"}:
        with fits.open(path, memmap=True) as hd:
            data = hd[0].data[y0:y1]
            arr = np.asarray(data, dtype=np.float32)
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
        libc.madvise(ctypes.c_void_p(int(mmap_obj.ctypes.data)), mmap_obj.nbytes, MADV_DONTNEED)
    except Exception:
        pass


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = read_paths(args.csv)
    if not files:
        print("CSV is empty", file=sys.stderr)
        return 1
    first = files[0]
    slice0 = open_slice(first, 0, 1)
    H = slice0.shape[0] * 1
    W = slice0.shape[1]
    C = slice0.shape[2]

    sum_path = os.path.join(args.out, "cum_sum.npy")
    wht_path = os.path.join(args.out, "cum_wht.npy")
    cum_sum = np.memmap(sum_path, mode="w+", dtype=np.float32, shape=(H, W, C))
    cum_sum[:] = 0
    cum_wht = np.memmap(wht_path, mode="w+", dtype=np.float32, shape=(H, W))
    cum_wht[:] = 1

    tile_h = int(args.tile)
    nfiles = len(files)
    for y0 in range(0, H, tile_h):
        y1 = min(y0 + tile_h, H)
        tile_slices = []
        for fp in files:
            tile_slices.append(open_slice(fp, y0, y1))
        tile = np.stack(tile_slices, axis=0)
        winsorize(tile, args.kappa, args.winsor)
        cum_sum[y0:y1] += tile.sum(axis=0)
        cum_wht[y0:y1] += tile.shape[0]
        del tile_slices
        del tile
        gc.collect()
        flush_mmap(cum_sum)
        flush_mmap(cum_wht)
        progress = 100.0 * y1 / H
        print(f"{progress:.1f}%", flush=True)

    final = cum_sum / np.maximum(cum_wht[..., None], 1e-6)
    fits.writeto(os.path.join(args.out, "final.fits"), final.astype(np.float32), overwrite=True)
    flush_mmap(cum_sum)
    flush_mmap(cum_wht)
    del cum_sum
    del cum_wht
    os.remove(sum_path)
    os.remove(wht_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc(limit=3)
        sys.exit(1)
