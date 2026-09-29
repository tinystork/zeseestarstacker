"""Lot B synthetic bounded GPU probe (run with the CuPy runtime, MX150).

Imports the repo checkout (sys.path[0]) and verifies, on the real 2 GiB MX150:

1. the TILED path never materialises a full host cube (np.stack interception);
2. the masks interface is bitwise-identical to the masked-input path and the
   untiled twin, for N in {1,2,3,5,20}, mono + RGB, NaN + masks, weights,
   band + rectangular tiles, and a small-N outlier;
3. the global z coordination (two-region witness) is preserved.

No M74 data, no heavy run.  Prints PASS/FAIL per check and exits nonzero on any
failure.
"""
from __future__ import annotations

import os
import sys

REPO = "/home/tristan/.openclaw/workspace/projects/zeseestarstacker"
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

import seestar.core.stack_gpu as sgp  # noqa: E402
from seestar.core.stack_gpu import (  # noqa: E402
    stack_winsorized_sigma_gpu,
    stack_winsorized_sigma_gpu_tiled,
)

DEFAULT_LIMITS = (0.05, 0.05)
failures = []


def check(tag, fn):
    try:
        fn()
        print("PASS %s" % tag)
    except Exception as exc:  # noqa: BLE001
        failures.append((tag, exc))
        print("FAIL %s: %r" % (tag, exc))


def _make_stack(n, shape, channels=None, seed=0, nan_frac=0.05, spike=0.06):
    rng = np.random.default_rng(seed)
    out_shape = (n,) + shape + ((channels,) if channels else ())
    a = rng.normal(1000.0, 20.0, size=out_shape).astype(np.float32)
    a[rng.random(out_shape) < nan_frac] = np.nan
    a = a + np.where(rng.random(out_shape) < spike, 400.0, 0.0).astype(np.float32)
    return a


def _weights(n):
    rng = np.random.default_rng(7)
    return rng.uniform(0.4, 1.6, size=n).astype(np.float32)


def _images(a):
    return [a[i] for i in range(a.shape[0])]


def _raw_masks(a):
    """Split a NaN-masked stack into raw images (NaN -> 0) + 2-D validity
    maps (True where the original had a finite sample).  Matches the
    pipeline's HW-bool mask contract: 2-D maps, full-pixel NaN only."""
    masks, raw = [], []
    for i in range(a.shape[0]):
        im = a[i]
        if im.ndim == 3:
            valid = ~np.isnan(im).any(axis=-1)
        else:
            valid = ~np.isnan(im)
        masks.append(valid)
        raw.append(np.where(np.isnan(im), np.float32(0.0), im))
    return raw, masks


# --- 1. no full host cube (np.stack interception) ---------------------------
def probe_no_full_cube():
    n, H, W = 30, 40, 96
    a = _make_stack(n, (H, W), seed=910)
    w = _weights(n)
    shapes = []
    real_stack = sgp.np.stack

    def spy(arrays, *args, **kwargs):
        out = real_stack(arrays, *args, **kwargs)
        shapes.append(out.shape)
        return out

    sgp.np.stack = spy
    try:
        stack_winsorized_sigma_gpu_tiled(_images(a), w, return_weights=True, tile_shape=16)
    finally:
        sgp.np.stack = real_stack
    assert shapes, "no np.stack calls recorded"
    for s in shapes:
        spatial = int(np.prod(s[1:])) if len(s) > 1 else 0
        assert spatial <= 16 * W, s
        assert spatial < H * W, s


# --- 2. masks interface bitwise parity -------------------------------------
def probe_masks_parity_mono():
    for n in (1, 2, 3, 5, 20):
        H, W = 40, 96
        a = _make_stack(n, (H, W), seed=1000 + n)
        if n <= 19:
            a[0, H // 2, W // 2] = 1e6  # gross outlier for the guard
        w = _weights(n)
        raw, masks = _raw_masks(a)
        full = stack_winsorized_sigma_gpu(_images(a), w, return_weights=True)
        for ts in (16, (8, 16)):
            ref = stack_winsorized_sigma_gpu_tiled(
                _images(a), w, return_weights=True, tile_shape=ts
            )
            got = stack_winsorized_sigma_gpu_tiled(
                raw, w, return_weights=True, tile_shape=ts, masks=masks
            )
            assert np.array_equal(got[0], ref[0], equal_nan=True), (n, ts)
            assert np.array_equal(got[1], ref[1]), (n, ts)
            assert got[2] == ref[2] == full[2], (n, ts)


def probe_masks_parity_rgb():
    n = 5
    H, W = 4, 48
    a = np.full((n, H, W, 3), 0.04, dtype=np.float32)
    a[0, :, :, :] = 0.0401
    a[1, :, :, :] = 0.0399
    a[2, :, :, :] = 0.0402
    a[4, :, :, :] = 0.9978
    raw, masks = _raw_masks(a)
    full = stack_winsorized_sigma_gpu(_images(a), None, return_weights=True)
    assert full[2] == 20.0
    for ts in (1, 2):
        ref = stack_winsorized_sigma_gpu_tiled(
            _images(a), None, return_weights=True, tile_shape=ts
        )
        got = stack_winsorized_sigma_gpu_tiled(
            raw, None, return_weights=True, tile_shape=ts, masks=masks
        )
        assert np.array_equal(got[0], ref[0], equal_nan=True), ts
        assert np.array_equal(got[1], ref[1]), ts
        assert got[2] == full[2], ts


# --- 3. global z coordination (two-region witness) -------------------------
def _two_region(n=30, seed=15):
    def clipped_gauss(rng, c=30.0):
        x = rng.normal(1000.0, 20.0, size=(n, 48, 40))
        return np.clip(x, 1000.0 - c, 1000.0 + c).astype(np.float32)

    def multi_outlier(rng):
        x = rng.normal(1000.0, 20.0, size=(n, 48, 40))
        for amp, frac in [(70, 0.05), (90, 0.04), (120, 0.03), (160, 0.02),
                          (250, 0.015), (400, 0.01)]:
            x = x + np.where(rng.random((n, 48, 40)) < frac, amp, 0.0)
        return x.astype(np.float32)

    rng = np.random.default_rng(seed)
    return np.concatenate([clipped_gauss(rng), multi_outlier(rng)], axis=1)


def probe_global_z():
    a = _two_region()
    raw, masks = _raw_masks(a)
    full = stack_winsorized_sigma_gpu(_images(a), None, return_weights=True)
    for ts in (48, 16, 7, 1):
        ref = stack_winsorized_sigma_gpu_tiled(
            _images(a), None, return_weights=True, tile_shape=ts
        )
        got = stack_winsorized_sigma_gpu_tiled(
            raw, None, return_weights=True, tile_shape=ts, masks=masks
        )
        assert np.array_equal(got[0], ref[0], equal_nan=True), ts
        assert np.array_equal(got[1], ref[1]), ts
        assert got[2] == ref[2] == full[2], ts


if __name__ == "__main__":
    import cupy as cp  # noqa: E402

    props = cp.cuda.runtime.getDeviceProperties(0)
    print("device: %s (vram %d MiB)" % (
        props["name"].decode(), props["totalGlobalMem"] // (1024 * 1024)))
    check("no_full_host_cube", probe_no_full_cube)
    check("masks_parity_mono_N1_2_3_5_20", probe_masks_parity_mono)
    check("masks_parity_rgb_small_n_outlier", probe_masks_parity_rgb)
    check("global_z_coordination", probe_global_z)
    if failures:
        print("\n%d FAILURE(S)" % len(failures))
        sys.exit(1)
    print("\nALL PROBES PASSED")
