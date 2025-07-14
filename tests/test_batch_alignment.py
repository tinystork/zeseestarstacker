import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seestar.core.batch_alignment import align_batch_to_reference


def make_star_field(size=40, num_stars=20, rng=None):
    rng = rng or np.random.default_rng(0)
    img = rng.normal(0, 0.01, size=(size, size)).astype(np.float32)
    pts = rng.integers(5, size - 5, size=(num_stars, 2))
    for x, y in pts:
        img[y, x] = 1.0
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def test_align_batch_to_reference():
    ref = make_star_field()
    M = cv2.getRotationMatrix2D((20, 20), 5, 1.0)
    M[0, 2] += 2.5
    M[1, 2] -= 3.0
    src = cv2.warpAffine(ref, M, (40, 40))
    aligned = align_batch_to_reference(src, ref)
    assert aligned.shape == ref.shape
