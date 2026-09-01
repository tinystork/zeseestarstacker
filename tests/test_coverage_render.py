"""COV-04 focused tests: final-only coverage-aware render.

The render only attenuates the high-frequency residual in low-support regions;
it must never brighten low-coverage pixels, invent signal, or modify the
low-frequency background.
"""

import numpy as np
import pytest

from seestar.enhancement.coverage_render import coverage_aware_render


def test_flat_field_unchanged_no_brightness_gain():
    # constant field stays constant regardless of support -> no gain
    H = W = 64
    sci = np.full((H, W), 1000.0, np.float32)
    sup = np.zeros((H, W), np.float32)
    sup[10:54, 10:54] = 100.0
    out = coverage_aware_render(sci, sup, n_ref=32.0)
    assert np.allclose(out, 1000.0, atol=1e-3)


def test_high_support_untouched():
    rng = np.random.default_rng(0)
    H = W = 64
    sci = rng.normal(50.0, 5.0, (H, W)).astype(np.float32)
    sup = np.full((H, W), 1000.0, np.float32)  # everywhere highly supported
    out = coverage_aware_render(sci, sup, n_ref=32.0)
    assert np.allclose(out, sci, atol=1e-4)


def test_low_support_denoised_mean_preserved():
    rng = np.random.default_rng(1)
    H = W = 96
    base = np.full((H, W), 100.0, np.float32)
    noise = rng.normal(0.0, 20.0, (H, W)).astype(np.float32)
    sci = base + noise
    sup = np.zeros((H, W), np.float32)
    sup[16:80, 16:80] = 4.0   # low support (<< n_ref=32)
    out = coverage_aware_render(sci, sup, n_ref=32.0, sigma_denoise=2.0)
    region = (slice(16, 80), slice(16, 80))
    # mean preserved (no brightness gain)
    assert abs(float(np.mean(out[region])) - 100.0) < 5.0
    # variance reduced (noise regularized)
    assert float(np.var(out[region])) < float(np.var(sci[region]))


def test_no_support_noop():
    sci = np.full((16, 16), 7.0, np.float32)
    out = coverage_aware_render(sci, np.zeros((16, 16), np.float32))
    assert np.array_equal(out, sci)


def test_shape_mismatch_rejected():
    sci = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError):
        coverage_aware_render(sci, np.zeros((4, 4), np.float32))


def test_color_unchanged_shape():
    rng = np.random.default_rng(2)
    sci = rng.normal(10.0, 2.0, (32, 32, 3)).astype(np.float32)
    sup = np.full((32, 32), 500.0, np.float32)
    out = coverage_aware_render(sci, sup)
    assert out.shape == sci.shape
    assert out.dtype == np.float32
