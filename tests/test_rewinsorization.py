import numpy as np
from zemosaic.zemosaic_align_stack import _reject_outliers_winsorized_sigma_clip


def test_rewinsor_replace_values():
    stack = np.stack([
        np.zeros((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32) * 10,
        np.ones((2, 2), dtype=np.float32) * 10,
        np.ones((2, 2), dtype=np.float32) * 10,
    ], axis=0)
    out, _ = _reject_outliers_winsorized_sigma_clip(
        stack, (0.25, 0.25), 1.0, 1.0, apply_rewinsor=True
    )
    assert not np.isnan(out).any()
    assert np.allclose(out[0], 10.0)


def test_rewinsor_nan_when_disabled():
    stack = np.stack([
        np.zeros((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32) * 10,
        np.ones((2, 2), dtype=np.float32) * 10,
        np.ones((2, 2), dtype=np.float32) * 10,
    ], axis=0)
    out, _ = _reject_outliers_winsorized_sigma_clip(
        stack, (0.25, 0.25), 1.0, 1.0, apply_rewinsor=False
    )
    assert np.isnan(out[0]).all()
