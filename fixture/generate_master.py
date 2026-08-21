"""Deterministic generator for the IMMUTABLE master reference dataset.

Produces ``fixture/master/Light_001.fit`` .. ``Light_010.fit`` — a small,
reproducible M16-like dataset used as the differential-oracle source for the
reliability audit (Tk vs Qt vs engine, same FITS, same settings, same engine).

Design constraints (mission ZSSS-QT-RELIABILITY-AUDIT):

* Realistic Seestar S50 geometry: ~1080x1920 (H x W) sensor, ~2.37 arcsec/px.
* Realistic TAN WCS centred on M16 (Eagle Nebula), CDELT ~2.37 arcsec/px.
* 16-bit unsigned integer data (Seestar-style RAW), smooth and NO destructive
  noise, so the files stay small in git (highly compressible) and fully
  deterministic.
* Fixed seed -> byte-reproducible across machines. MASTER IS NEVER MODIFIED.
  If the spec must change, bump a version suffix and regenerate explicitly.

Run:  python fixture/generate_master.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

MASTER_DIR = Path(__file__).resolve().parent / "master"

# Geometry (Seestar S50)
HEIGHT = 1080
WIDTH = 1920
N_IMAGES = 10
PIXEL_SCALE_ARCSEC = 2.37
PIXEL_SCALE_DEG = PIXEL_SCALE_ARCSEC / 3600.0

# M16 (Eagle Nebula) centre
RA_DEG = 274.7499
DEC_DEG = -13.8200

SEED = 20260821

# Per-frame dither offsets in arcsec (small, deterministic, non-destructive).
_DITHER_ARCSEC = [
    (0.00, 0.00),
    (0.31, -0.47),
    (-0.52, 0.23),
    (0.74, 0.41),
    (-0.19, -0.38),
    (0.58, -0.11),
    (-0.68, 0.82),
    (0.12, 0.53),
    (-0.43, -0.71),
    (0.95, 0.08),
]


def make_wcs(shape_hw, crval_ra, crval_dec, cdelt_deg):
    h, w = shape_hw
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    wcs.wcs.crval = [crval_ra, crval_dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.cdelt = [-cdelt_deg, cdelt_deg]
    wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
    wcs.array_shape = (h, w)
    return wcs


def _gauss(shape_hw, amp, sig_px, centre):
    h, w = shape_hw
    yy, xx = np.ogrid[:h, :w]
    return amp * np.exp(
        -((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / (2.0 * sig_px**2)
    )


def build_frame(shape_hw, seed, dither_arcsec):
    """Smooth deterministic nebula gradient + a handful of soft stars.

    Returns a float64 HxW frame in [0, 1]; the caller converts to uint16.
    """
    h, w = shape_hw
    rng = np.random.default_rng(seed)

    # Broad smooth nebula gradient (compressible, deterministic).
    yy, xx = np.ogrid[:h, :w]
    nebula = (
        0.10
        + 0.25 * np.exp(-((xx - w * 0.62) ** 2 + (yy - h * 0.45) ** 2) / (2.0 * (w * 0.22) ** 2))
        + 0.18 * np.exp(-((xx - w * 0.35) ** 2 + (yy - h * 0.60) ** 2) / (2.0 * (w * 0.30) ** 2))
    )

    # A fixed star field: N stars at deterministic positions/sizes/brightnesses.
    n_stars = 32
    star_x = rng.uniform(0, w, n_stars)
    star_y = rng.uniform(0, h, n_stars)
    star_amp = rng.uniform(0.15, 0.95, n_stars)
    star_sig = rng.uniform(1.2, 2.8, n_stars)

    # Apply the dither as a sub-pixel translation (arcsec -> px).
    dx_px = dither_arcsec[0] / PIXEL_SCALE_ARCSEC
    dy_px = dither_arcsec[1] / PIXEL_SCALE_ARCSEC

    frame = nebula.copy()
    for sx, sy, amp, sig in zip(star_x, star_y, star_amp, star_sig):
        frame += _gauss(shape_hw, amp, sig, (sx + dx_px, sy + dy_px))

    # Soft clip to [0, 1] and de-emphasise the floor (no destructive noise).
    return np.clip(frame, 0.0, 1.0)


def generate(master_dir: Path | None = None) -> list[Path]:
    master_dir = Path(master_dir) if master_dir else MASTER_DIR
    master_dir.mkdir(parents=True, exist_ok=True)

    shape_hw = (HEIGHT, WIDTH)
    written = []
    for i in range(N_IMAGES):
        frame = build_frame(shape_hw, SEED + i, _DITHER_ARCSEC[i % len(_DITHER_ARCSEC)])
        # Seestar-style 16-bit RAW.
        data = (frame * 65535.0).astype(np.uint16)

        wcs = make_wcs(shape_hw, RA_DEG, DEC_DEG, PIXEL_SCALE_DEG)
        hdr = wcs.to_header()
        hdr["EXPTIME"] = (10.0, "exposure seconds (synthetic)")
        hdr["OBJECT"] = ("M16", "Eagle Nebula (synthetic)")
        hdr["INSTRUME"] = ("SYNTH", "synthetic reference dataset")
        hdr["BUNIT"] = ("ADU", "analog-to-digital units")
        hdr["SEED"] = (SEED + i, "generation seed")

        path = master_dir / f"Light_{i + 1:03d}.fit"
        fits.writeto(path, data, header=hdr, overwrite=True)
        written.append(path)

    return written


if __name__ == "__main__":
    paths = generate()
    print(f"Wrote {len(paths)} master FITS to {MASTER_DIR}:")
    for p in paths:
        print(f"  {p.name}  ({os.path.getsize(p)} bytes)")
