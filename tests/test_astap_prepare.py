import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zemosaic.zemosaic_worker import _prepare_image_for_astap


def test_mono_stays_2d():
    arr = np.ones((5, 5), dtype=np.float32)
    out = _prepare_image_for_astap(arr)
    assert out.ndim == 2


def test_hwc_to_chw():
    arr = np.zeros((4, 6, 3), dtype=np.float32)
    out = _prepare_image_for_astap(arr)
    assert out.shape == (3, 4, 6)


def test_chw_stays_chw():
    arr = np.zeros((3, 4, 6), dtype=np.float32)
    out = _prepare_image_for_astap(arr)
    assert out.shape == (3, 4, 6)


def test_force_lum():
    arr = np.zeros((3, 4, 6), dtype=np.float32)
    out = _prepare_image_for_astap(arr, force_lum=True)
    assert out.ndim == 2 and out.shape == (4, 6)
