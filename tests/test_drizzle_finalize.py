import importlib.util
import sys
from pathlib import Path

import numpy as np
from drizzle.resample import Drizzle

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
    pixmap = np.dstack(np.indices((16, 16))[::-1]).astype(np.float32)
    d_single = Drizzle(out_shape=(16, 16))
    for f in frames:
        d_single.add_image(f, exptime=1.0, pixmap=pixmap)
    ref = drizzle_finalize(d_single.out_img, d_single.out_wht)

    def run_batch(fs):
        d = Drizzle(out_shape=(16, 16))
        for f in fs:
            d.add_image(f, exptime=1.0, pixmap=pixmap)
        return d.out_img, d.out_wht

    s1, w1 = run_batch(frames[:10])
    s2, w2 = run_batch(frames[10:])
    test = drizzle_finalize(s1 + s2, w1 + w2)

    star = (8, 8)
    assert abs(ref[star] - test[star]) / ref[star] < 1.1
    med_ref, med_test = np.median(ref), np.median(test)
    p99_ref, p99_test = np.percentile(ref, 99), np.percentile(test, 99)
    assert abs(med_ref - med_test) / med_ref <= 1.0
    assert abs(p99_ref - p99_test) / p99_ref <= 1.0
