import importlib.util
import sys
from pathlib import Path

import numpy as np
from astropy.wcs import WCS
from seestar.enhancement.drizzle_integration import run_incremental_drizzle

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "drizzle_utils", ROOT / "seestar" / "core" / "drizzle_utils.py"
)
drizzle_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drizzle_utils)
drizzle_finalize = drizzle_utils.drizzle_finalize


def make_star_frame(shape=(16, 16), flux=1000.0, pos=(8, 8)):
    img = np.ones(shape, dtype=np.float32)
    img[pos] += flux
    return img


def test_drizzle_double_norm_fix_consistency():
    frames = [make_star_frame() for _ in range(20)]
    w = WCS(naxis=2)
    w.wcs.crpix = [8, 8]
    w.wcs.cdelt = [1, 1]
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ref = run_incremental_drizzle(frames, [w]*len(frames), w, (16,16))

    def run_batch(fs):
        return run_incremental_drizzle(fs, [w]*len(fs), w, (16,16))

    s1 = run_batch(frames[:10])
    s2 = run_batch(frames[10:])
    test = drizzle_finalize(s1 + s2, np.ones_like(s1))

    star = (8, 8)
    assert abs(ref[star] - test[star]) / ref[star] < 1.1
    med_ref, med_test = np.median(ref), np.median(test)
    p99_ref, p99_test = np.percentile(ref, 99), np.percentile(test, 99)
    assert abs(med_ref - med_test) / med_ref <= 1.0
    assert abs(p99_ref - p99_test) / p99_ref <= 1.0
