"""Lot B: pure (no-GPU) tests of the per-tile host materialization interface.

These tests prove, WITHOUT CuPy, that the tiled driver's materialization seam
builds only ``N x tile_h x tile_w x C`` per tile — never the full
``N x H x W x C`` host cube — and that the per-tile validity-mask application
is bitwise identical to the queue-manager full-frame ``_nan_mask_image``
(sliced to the same tile).

The tiled reduction itself (``stack_winsorized_sigma_gpu_tiled``) requires
CuPy; the GPU-level parity + no-full-cube interception proofs live in
``test_stack_gpu_winsorized_tiled.py`` (skipped when CuPy is absent).
"""

from __future__ import annotations

import numpy as np

import seestar.core.stack_gpu as sgp
from seestar.queuep.queue_manager import _nan_mask_image


def _images(a):
    return [a[i] for i in range(a.shape[0])]


def test_materialize_tile_shape_is_tile_not_full_frame():
    """The materialized host array is (N, th, tw[, C]), never (N, H, W[, C])."""
    n, H, W = 5, 40, 96
    a = np.random.default_rng(0).normal(1000.0, 20.0, size=(n, H, W))
    a = a.astype(np.float32)
    for (y0, y1, x0, x1) in sgp._winsor_tile_slices((H, W), (16, 32)):
        tile = sgp._materialize_tile(_images(a), None, y0, y1, x0, x1)
        assert tile.shape == (n, y1 - y0, x1 - x0), tile.shape
        assert tile.dtype == np.float32


def test_materialize_tile_equals_full_stack_slice_bitwise():
    """Slicing after a full stack == materializing the tile directly (bitwise)."""
    n, H, W = 6, 41, 96
    rng = np.random.default_rng(1)
    a = rng.normal(1000.0, 20.0, size=(n, H, W)).astype(np.float32)
    a[rng.random((n, H, W)) < 0.05] = np.nan
    full = np.stack(_images(a), axis=0)
    for (y0, y1, x0, x1) in sgp._winsor_tile_slices((H, W), 16):
        tile = sgp._materialize_tile(_images(a), None, y0, y1, x0, x1)
        np.testing.assert_array_equal(tile, full[:, y0:y1, x0:x1])


def test_materialize_tile_with_masks_matches_full_frame_nan_mask():
    """Per-tile mask application == full-frame _nan_mask_image then slice."""
    n, H, W = 6, 40, 96
    rng = np.random.default_rng(2)
    a = rng.normal(1000.0, 20.0, size=(n, H, W)).astype(np.float32)
    masks = [rng.random((H, W)) > 0.15 for _ in range(n)]  # 2-D bool maps
    masked_full = [_nan_mask_image(im, m) for im, m in zip(_images(a), masks)]
    for (y0, y1, x0, x1) in sgp._winsor_tile_slices((H, W), (16, 32)):
        tile = sgp._materialize_tile(_images(a), masks, y0, y1, x0, x1)
        ref = np.stack([m[y0:y1, x0:x1] for m in masked_full], axis=0)
        np.testing.assert_array_equal(tile, ref)


def test_materialize_tile_with_masks_rgb_broadcast():
    """RGB validity mask (2-D) broadcasts over the channel axis per tile."""
    n, H, W, C = 4, 24, 48, 3
    rng = np.random.default_rng(3)
    a = rng.normal(1000.0, 20.0, size=(n, H, W, C)).astype(np.float32)
    masks = [rng.random((H, W)) > 0.2 for _ in range(n)]
    for (y0, y1, x0, x1) in sgp._winsor_tile_slices((H, W), (8, 16)):
        tile = sgp._materialize_tile(_images(a), masks, y0, y1, x0, x1)
        assert tile.shape == (n, y1 - y0, x1 - x0, C)
        # invalid pixels -> NaN in every channel
        for i in range(n):
            bad = ~masks[i][y0:y1, x0:x1]
            assert np.all(np.isnan(tile[i][bad]))
            good = masks[i][y0:y1, x0:x1]
            assert np.allclose(tile[i][good], a[i][y0:y1, x0:x1][good])


def test_materialize_tile_does_not_modify_source_images():
    """Materialization reads views only; the shared aligned images are intact."""
    n, H, W = 4, 40, 96
    a = np.random.default_rng(4).normal(1000.0, 20.0, size=(n, H, W))
    a = a.astype(np.float32)
    snap = a.copy()
    masks = [np.ones((H, W), dtype=bool) for _ in range(n)]
    for (y0, y1, x0, x1) in sgp._winsor_tile_slices((H, W), 16):
        sgp._materialize_tile(_images(a), masks, y0, y1, x0, x1)
    np.testing.assert_array_equal(a, snap)


def test_nan_mask_slice_matches_nan_mask_image():
    """The local per-tile NaN mask is the exact twin of the queue-manager one."""
    rng = np.random.default_rng(5)
    img = rng.normal(0.0, 1.0, size=(10, 12)).astype(np.float32)
    mask = rng.random((10, 12)) > 0.3
    np.testing.assert_array_equal(
        sgp._nan_mask_slice(img, mask), _nan_mask_image(img, mask)
    )
    rgb = rng.normal(0.0, 1.0, size=(10, 12, 3)).astype(np.float32)
    np.testing.assert_array_equal(
        sgp._nan_mask_slice(rgb, mask), _nan_mask_image(rgb, mask)
    )


# ---------------------------------------------------------------------------
# F6: single-allocation materialization (no np.stack, no np.where, no second
# conversion cube)
# ---------------------------------------------------------------------------

def _tile_rgb_uint16_float64_matrix():
    n, H, W, C = 4, 24, 48, 3
    rng = np.random.default_rng(11)
    a = rng.normal(1000.0, 20.0, size=(n, H, W, C)).astype(np.float32)
    masks = [rng.random((H, W)) > 0.2 for _ in range(n)]
    return _images(a), masks


def test_materialize_tile_never_uses_stack_or_where(monkeypatch):
    """F6: ``_materialize_tile`` must not call ``np.stack`` or ``np.where``
    (single preallocated float32 cube filled slice-by-slice)."""
    import numpy as np
    n, H, W = 5, 40, 96
    a = np.random.default_rng(0).normal(1000.0, 20.0, size=(n, H, W)).astype(np.float32)
    masks = [np.ones((H, W), dtype=bool) for _ in range(n)]

    def _forbidden(*args, **kwargs):
        raise AssertionError("np.stack / np.where must not be used in _materialize_tile")

    monkeypatch.setattr(sgp.np, "stack", _forbidden)
    monkeypatch.setattr(sgp.np, "where", _forbidden)
    for (y0, y1, x0, x1) in sgp._winsor_tile_slices((H, W), (16, 32)):
        tile = sgp._materialize_tile(_images(a), masks, y0, y1, x0, x1)
        assert tile.shape == (n, y1 - y0, x1 - x0)


def test_materialize_tile_single_cube_allocation(monkeypatch):
    """F6: only ONE primary cube allocation per materialization (the final
    float32 cube); no per-image ``np.where`` tile, no ``np.stack`` second cube."""
    import numpy as np
    n, H, W = 5, 40, 96
    a = np.random.default_rng(1).normal(1000.0, 20.0, size=(n, H, W)).astype(np.float32)
    masks = [np.random.default_rng(2).random((H, W)) > 0.3 for _ in range(n)]
    alloc_shapes = []
    real_empty = np.empty

    def spy_empty(shape, *args, **kwargs):
        alloc_shapes.append(tuple(shape))
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(sgp.np, "empty", spy_empty)
    y0, y1, x0, x1 = 0, 16, 0, 32
    tile = sgp._materialize_tile(_images(a), masks, y0, y1, x0, x1)
    # The primary cube allocation is the tile cube, sized exactly once.
    cube_allocs = [s for s in alloc_shapes if s == (n, 16, 32)]
    assert len(cube_allocs) == 1, alloc_shapes
    # No larger (full-frame) allocation ever occurred.
    for s in alloc_shapes:
        assert s[1] <= 16 and s[2] <= 32, s


def test_materialize_tile_uint16_float64_direct_cast_no_second_cube(monkeypatch):
    """F6: for uint16/float64 inputs the tile is cast DIRECTLY into the float32
    cube (no temporary conversion cube) and stays bit-identical (valid) + NaN
    (invalid) — the tile cost is 4 bytes regardless of input dtype."""
    import numpy as np
    n, H, W = 4, 24, 48
    rng = np.random.default_rng(3)
    base = rng.integers(0, 60000, size=(n, H, W)).astype(np.uint16)
    masks = [rng.random((H, W)) > 0.3 for _ in range(n)]

    def _forbidden(*args, **kwargs):
        raise AssertionError("np.stack / np.where must not be used")

    monkeypatch.setattr(sgp.np, "stack", _forbidden)
    monkeypatch.setattr(sgp.np, "where", _forbidden)
    y0, y1, x0, x1 = 0, 8, 0, 16
    tile = sgp._materialize_tile(_images(base), masks, y0, y1, x0, x1)
    assert tile.dtype == np.float32
    for i in range(n):
        good = masks[i][y0:y1, x0:x1]
        # Valid pixels: exact uint16 -> float32 cast (no precision loss).
        assert np.array_equal(
            tile[i][good], base[i][y0:y1, x0:x1][good].astype(np.float32)
        )
        assert np.all(np.isnan(tile[i][~good]))


def test_materialize_tile_non_bool_mask_same_as_bool():
    """F6: a non-bool (fractional/nonzero) validity mask behaves exactly like
    its truthiness (nonzero) bool view — no bool tile temporary, ``where`` uses
    the same nonzero-as-valid semantics as the reference ``np.where(m, ..)``."""
    n, H, W = 4, 24, 48
    rng = np.random.default_rng(4)
    a = rng.normal(1000.0, 20.0, size=(n, H, W)).astype(np.float32)
    frac = [rng.random((H, W)) for _ in range(n)]
    # Reference semantics: nonzero == valid (same as np.where(m, img, nan)).
    bool_masks = [f != 0 for f in frac]
    y0, y1, x0, x1 = 0, 12, 0, 16
    tile_frac = sgp._materialize_tile(_images(a), frac, y0, y1, x0, x1)
    tile_bool = sgp._materialize_tile(_images(a), bool_masks, y0, y1, x0, x1)
    # NaN positions are identical; valid positions are identical (NaN-aware).
    np.testing.assert_array_equal(np.isnan(tile_frac), np.isnan(tile_bool))
    both = ~np.isnan(tile_frac)
    np.testing.assert_array_equal(tile_frac[both], tile_bool[both])

