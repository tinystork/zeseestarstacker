"""
livestack_mode.py – Image‑by‑image LiveStack **with on‑the‑fly alignment**
======================================================================

* ZeSeestarStacker auxiliary module – May 2025
* Author : ChatGPT helper – generated for Gizmo

This controller performs **true Live‑Stacking**:

* Loads FITS/fit images sequentially from an input folder
* Optional **SNR rejection** (scalar global SNR)
* **Automatic reference selection** (first accepted image)
* Aligns each subsequent image on the reference using **astroalign**
* Accumulates into SUM/WHT mem‑maps (same format as QueueManager)
* Calls preview / progress callbacks after **every accepted image**
* Saves drizzle‑ready aligned FITS files in a temp Dir
* At the end – drizzle with `stsci.drizzle` then exports **PNG 16‑bit** +
  optional **FITS drizzled**
* Offers `save_current_stack()` helper so the GUI can add a "Save"
  button letting the user choose PNG / TIFF / FITS at any moment.

Designed to be self‑contained: drop it under `seestar/modes/` and call
from a dedicated "LiveStack" tab.
"""
from __future__ import annotations

import sys
import traceback
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
from astropy.io import fits
import cv2
import astroalign as aa

# stsci.drizzle imports
try:
    from drizzle.resample import Drizzle
except ImportError:
    Drizzle = None  # handled later

# === Type aliases
PreviewCB = Callable[[np.ndarray, fits.Header | None, str], None]
ProgressCB = Callable[[str], None]

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------

def compute_snr(img: np.ndarray) -> float:
    """Compute a very rough per‑frame SNR (median / MAD)."""
    if img is None:
        return 0.0
    finite = img[np.isfinite(img)]
    if finite.size < 100:
        return 0.0
    med = np.median(finite)
    mad = np.median(np.abs(finite - med)) * 1.4826 + 1e-6
    return float(med / mad)


def stretch_01(img: np.ndarray) -> np.ndarray:
    """Linear stretch to 0‑1 (keeps NaNs -> 0)."""
    if img is None:
        return img
    vmin, vmax = np.nanmin(img), np.nanmax(img)
    if vmax > vmin:
        out = (img - vmin) / (vmax - vmin)
    else:
        out = np.zeros_like(img)
    return np.clip(np.nan_to_num(out, 0.0), 0.0, 1.0)


def save_png16(path: Path, data: np.ndarray):
    """Save 0‑1 float32 RGB as 16‑bit PNG."""
    arr = (np.clip(data, 0.0, 1.0) * 65535.0).astype(np.uint16)
    if arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
    elif arr.ndim == 2:
        cv2.imwrite(str(path), arr)
    else:
        raise ValueError("Unsupported PNG shape: " + str(arr.shape))

# ----------------------------------------------------------------------
# LiveStackController
# ----------------------------------------------------------------------

class LiveStackController:
    """Image‑by‑image LiveStack with optional SNR rejection *and* alignment."""

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        snr_threshold: float = 2.5,
        align: bool = True,
        drizzle_scale: int = 2,
        preview_callback: Optional[PreviewCB] = None,
        progress_callback: Optional[ProgressCB] = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.snr_threshold = snr_threshold
        self.align = align
        self.preview_cb = preview_callback
        self.progress_cb = progress_callback
        self.drizzle_scale = int(drizzle_scale)

        # Runtime state
        self._stop_flag = False
        self.ref_image: Optional[np.ndarray] = None
        self.ref_header: Optional[fits.Header] = None
        self.ref_gray: Optional[np.ndarray] = None  # for correlation/astroalign
        self.accepted_count = 0
        self.rejected_count = 0

        # SUM/W memmaps created on first accepted image
        self.sum_mem: Optional[np.memmap] = None
        self.wht_mem: Optional[np.memmap] = None
        self.mem_shape: Optional[Tuple[int, int, int]] = None
        self.mem_dir = self.output_dir / "memmap"
        self.drizzle_temp_dir = self.output_dir / "drizzle_temp_inputs"
        self.drizzle_temp_dir.mkdir(parents=True, exist_ok=True)
        self.mem_dir.mkdir(parents=True, exist_ok=True)

        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input dir {self.input_dir} not found")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------- public API

    def stop(self):
        self._stop_flag = True

    def run_blocking(self):
        """Runs in current thread (for CLI)."""
        self._log("🚀 LiveStack started…")
        try:
            self._process_all()
            self._finalize_stack()
        except Exception as exc:
            self._log(f"❌ LiveStack aborted: {exc}")
            traceback.print_exc()
        self._log("🏁 LiveStack finished")

    def run_threaded(self):
        threading.Thread(target=self.run_blocking, daemon=True).start()

    # Called from GUI button
    def save_current_stack(self, filepath: str | Path, fmt: str = "png") -> None:
        if self.sum_mem is None or self.wht_mem is None:
            self._log("⚠️ No stack yet – nothing to save.")
            return
        filepath = Path(filepath)
        stack = self._current_average()
        if stack is None:
            self._log("⚠️ Stack empty.")
            return
        if fmt.lower() == "png":
            save_png16(filepath, stretch_01(stack))
        elif fmt.lower() in {"tiff", "tif"}:
            arr16 = (np.clip(stretch_01(stack), 0.0, 1.0) * 65535).astype(np.uint16)
            bgr = cv2.cvtColor(arr16, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filepath), bgr)
        elif fmt.lower() == "fits":
            hdr = self.ref_header if self.ref_header else fits.Header()
            fits.writeto(filepath, np.moveaxis(stack.astype(np.float32), -1, 0), header=hdr, overwrite=True)
        else:
            raise ValueError("Unknown format: " + fmt)
        self._log(f"💾 Saved stack → {filepath}")

    # -------------------------------------------------- internal

    def _process_all(self):
        for idx, path in enumerate(sorted(self._list_fits())):
            if self._stop_flag:
                break
            img, hdr = self._load(path)
            if img is None:
                self._log(f"⏭️ {path.name}: load failed")
                continue
            snr = compute_snr(img)
            if snr < self.snr_threshold:
                self.rejected_count += 1
                self._log(f"❌ {path.name}: SNR {snr:.2f} < {self.snr_threshold}")
                continue

            if self.ref_image is None:
                self._init_reference(img, hdr)
                aligned = img
            else:
                aligned = self._align(img) if self.align else img
                if aligned is None:
                    self._log(f"⚠️ {path.name}: alignment failed, skipped")
                    self.rejected_count += 1
                    continue

            self._accumulate(aligned)
            self._save_drizzle_temp(aligned, hdr, self.accepted_count)
            self.accepted_count += 1
            self._preview()

    # -------------------------------------------------- helpers

    def _list_fits(self) -> List[Path]:
        return [p for p in self.input_dir.iterdir() if p.suffix.lower() in {".fits", ".fit"}]

    def _load(self, path: Path) -> Tuple[Optional[np.ndarray], Optional[fits.Header]]:
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data.astype(np.float32)
                hdr = hdul[0].header
            # normalize 0‑1 quick
            data = stretch_01(data)
            # ensure H,W,C
            if data.ndim == 2:
                data = cv2.cvtColor((data*255).astype(np.uint8), cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            elif data.ndim == 3 and data.shape[0] == 3:
                data = np.moveaxis(data, 0, -1)  # C,H,W -> H,W,C
            return data, hdr
        except Exception as exc:
            self._log(f"Error loading {path.name}: {exc}")
            return None, None

    def _init_reference(self, img: np.ndarray, hdr: fits.Header):
        self.ref_image = img.copy()
        self.ref_header = hdr.copy() if hdr else fits.Header()
        # grayscale for alignment
        self.ref_gray = cv2.cvtColor((stretch_01(img)*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        h, w, _ = img.shape
        self.mem_shape = (h, w, 3)
        self.sum_mem = np.lib.format.open_memmap(self.mem_dir / "SUM.npy", mode="w+", dtype=np.float32, shape=self.mem_shape)
        self.wht_mem = np.lib.format.open_memmap(self.mem_dir / "WHT.npy", mode="w+", dtype=np.float32, shape=(h, w))
        self.sum_mem[:] = 0.0
        self.wht_mem[:] = 0.0
        self._log(f"🔰 Reference set ({h}×{w})")

    def _align(self, img: np.ndarray) -> Optional[np.ndarray]:
        try:
            gray = cv2.cvtColor((stretch_01(img)*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            transf, _ = aa.find_transform(gray, self.ref_gray)
            aligned16 = aa.apply_transform(transf, img, self.ref_image.shape)
            return aligned16.astype(np.float32)
        except aa.MaxIterError:
            return None
        except Exception as exc:
            self._log(f"align error: {exc}")
            return None

    def _accumulate(self, img: np.ndarray):
        if self.sum_mem is None:
            return
        self.sum_mem += img.astype(np.float32)
        self.wht_mem += 1.0

    def _current_average(self) -> Optional[np.ndarray]:
        if self.sum_mem is None:
            return None
        wht = np.maximum(self.wht_mem, 1e-6)
        avg = self.sum_mem / wht[..., None]
        return np.nan_to_num(avg, 0.0)

    def _save_drizzle_temp(self, img: np.ndarray, hdr: fits.Header, idx: int):
        path = self.drizzle_temp_dir / f"dz_{idx:04d}.fits"
        data = np.moveaxis(img.astype(np.float32), -1, 0)  # C,H,W
        try:
            fits.writeto(path, data, header=hdr, overwrite=True)
        except Exception as exc:
            self._log(f"Could not write drizzle temp {path.name}: {exc}")

    def _preview(self):
        if self.preview_cb is None:
            return
        avg = self._current_average()
        if avg is not None:
            self.preview_cb(avg, None, f"LiveStack ({self.accepted_count} img)")

    def _finalize_stack(self):
        if self.accepted_count == 0:
            self._log("No images accepted – nothing to finalize")
            return

        # Drizzle only if library available
        if Drizzle is None:
            self._log("stsci.drizzle not installed – skipping drizzle export")
            return

        self._log("☕ Drizzling …")
        temp_files = sorted(self.drizzle_temp_dir.glob("dz_*.fits"))
        if not temp_files:
            self._log("No temp drizzle inputs found!")
            return

        # load first file to set output grid
        sample_data, sample_hdr = self._load(temp_files[0])
        h, w, _ = sample_data.shape
        out_h, out_w = int(h * self.drizzle_scale), int(w * self.drizzle_scale)
        out_img = np.zeros((out_h, out_w, 3), dtype=np.float32)
        out_wht = np.zeros((out_h, out_w, 3), dtype=np.float32)
        # Dummy WCS (pixel scale only)
        out_wcs = None

        drizzlers = [Drizzle(out_img[..., c], out_wht[..., c], (out_h, out_w), out_wcs, kernel="square", pixfrac=1.0) for c in range(3)]

        for f in temp_files:
            data, _ = self._load(f)
            if data is None:
                continue
            for c in range(3):
                drizzlers[c].add_image(data[..., c], wcs=None, exposure=1.0)
        # fetch output
        final = np.dstack([d.outsci.astype(np.float32) for d in drizzlers])
        final = stretch_01(final)
        png_path = self.output_dir / "livestack_drizzled.png"
        save_png16(png_path, final)
        self._log(f"✅ PNG drizzled saved → {png_path.relative_to(self.output_dir)}")

        fits_path = self.output_dir / "livestack_drizzled.fits"
        fits.writeto(fits_path, np.moveaxis(final.astype(np.float32), -1, 0), overwrite=True)
        self._log(f"✅ FITS drizzled saved → {fits_path.relative_to(self.output_dir)}")

    # -------------------------------------------------- logging utilities

    def _log(self, msg: str):
        if self.progress_cb:
            self.progress_cb(msg)
        else:
            print(msg)

# ----------------------------------------------------------------------
# Convenience wrapper
# ----------------------------------------------------------------------

def start_livestack_cli(input_dir: str, output_dir: str, snr: float = 2.5):
    ctrl = LiveStackController(input_dir, output_dir, snr_threshold=snr)
    ctrl.run_blocking()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python livestack_mode.py <input_dir> <output_dir> [snr]")
        sys.exit(1)
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    snr_thr = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
    start_livestack_cli(in_dir, out_dir, snr_thr)
