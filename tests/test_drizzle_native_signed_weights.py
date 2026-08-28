"""Focused tests for native STScI Drizzle output / signed-weight closure (ZSSS-DNOW-01 R1).

Proves the M3 ``DrizzleAccumulator`` finalization:

* preserves native ``out_img`` science directly on valid (finite, positive
  ``wht > WEIGHT_EPSILON``) support;
* maps zero / near-zero / negative / non-finite native WHT — and non-finite
  native science — to ``0.0``, never a huge finite value;
* never mutates the caller / upstream accumulation arrays;
* keeps the square kernel output materially unchanged vs. the legacy
  ``(out_img * out_wht) / max(wht, 1e-9)`` formula on positive support;
* reproduces a real signed-Lanczos WHT but produces no artificial 1e10 values;
* wires the Lanczos policy (effective pixfrac 1.0, effective WHT threshold 0.0)
  and the square no-op policy at the queue_manager accumulator boundary;
* routes the live preview through the same native-safe ``finalize("divide")``.

All tests are small and deterministic.  The real signed-Lanczos probe uses a
sub-pixel-shifted partial-coverage frame, which reproduces negative ``out_wht``
lobes in the installed ``drizzle`` 2.2.0 engine.
"""

import numpy as np
import pytest
from astropy.wcs import WCS

from seestar.core.drizzle_core import (
    DrizzleAccumulator,
    LANCZOS_KERNELS,
    WEIGHT_EPSILON,
    support_integrity_violations,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _legacy_finalize_divide(acc):
    """Replicate the OLD (buggy) ``finalize("divide")`` formula verbatim.

    ``sci = out_img * out_wht`` (float32), then ``sci / max(wht, 1e-9)`` with
    ``nan_to_num``.  Used only to prove the square kernel output is materially
    unchanged on positive support and that the signed-WHT case was the bug.
    """
    sci = (acc._out_img * acc._out_wht).astype(np.float32)
    wht = acc._out_wht.astype(np.float32)
    wht_safe = np.maximum(wht, 1e-9)
    return np.nan_to_num(sci / wht_safe, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def _diff_metrics(new, ref):
    diff = np.abs(np.asarray(new, dtype=np.float64) - np.asarray(ref, dtype=np.float64))
    flat = diff.ravel()
    finite = np.isfinite(flat)
    flat = flat[finite]
    differing = int(np.count_nonzero(flat > 1e-6))
    total = int(flat.size)
    return {
        "max_abs": float(np.max(flat)) if flat.size else 0.0,
        "rms": float(np.sqrt(np.mean(flat ** 2))) if flat.size else 0.0,
        "median_abs": float(np.median(flat)) if flat.size else 0.0,
        "p99_9": float(np.percentile(flat, 99.9)) if flat.size else 0.0,
        "differing_count": differing,
        "differing_fraction": (differing / total) if total else 0.0,
    }


def make_wcs(shape_hw, crval=(10.0, 20.0), cdelt=(-0.001, 0.001)):
    h, w = shape_hw
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = np.array(list(cdelt))
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
    wcs.array_shape = (h, w)
    return wcs


# --------------------------------------------------------------------------
# 1-6. direct finalize unit tests (no engine required)
# --------------------------------------------------------------------------


def test_positive_wht_preserves_native_out_img():
    acc = DrizzleAccumulator((8, 8))
    native = (np.arange(64, dtype=np.float32).reshape(8, 8) * 1.5 + 10.0)
    acc._out_img[:] = native
    acc._out_wht[:] = 2.5
    out = acc.finalize("divide")
    assert out.dtype == np.float32
    # exact preservation of the native weighted-mean science
    assert np.array_equal(out, native)
    # upstream buffers untouched
    assert np.array_equal(acc._out_img, native)
    assert np.all(acc._out_wht == 2.5)


def test_zero_wht_maps_to_zero():
    acc = DrizzleAccumulator((8, 8))
    acc._out_img[:] = 123.0
    acc._out_wht[:] = 0.0
    assert np.all(acc.finalize("divide") == 0.0)


def test_near_zero_positive_wht_maps_to_zero():
    acc = DrizzleAccumulator((8, 8))
    acc._out_img[:] = 123.0
    # below epsilon
    acc._out_wht[:] = WEIGHT_EPSILON * 0.5
    assert np.all(acc.finalize("divide") == 0.0)
    # exactly epsilon is also invalid (strictly > epsilon required)
    acc._out_wht[:] = WEIGHT_EPSILON
    assert np.all(acc.finalize("divide") == 0.0)


def test_negative_wht_maps_to_zero_and_never_huge():
    acc = DrizzleAccumulator((8, 8))
    acc._out_img[:] = 500.0
    acc._out_wht[:] = 1.0
    acc._out_wht[0, 0] = -0.133
    acc._out_wht[1, 1] = -3.0
    out = acc.finalize("divide")
    assert out[0, 0] == 0.0
    assert out[1, 1] == 0.0
    assert np.all(np.isfinite(out))
    # no huge finite value anywhere (the old formula produced ~1e10 here)
    assert float(np.max(np.abs(out))) <= 500.0


def test_nonfinite_wht_and_science_map_to_finite_zero():
    acc = DrizzleAccumulator((8, 8))
    acc._out_img[:] = 1.0
    acc._out_wht[:] = 1.0
    acc._out_wht[2, 2] = np.nan
    acc._out_wht[3, 3] = np.inf
    acc._out_img[4, 4] = np.nan
    acc._out_img[5, 5] = np.inf
    out = acc.finalize("divide")
    for yx in ((2, 2), (3, 3), (4, 4), (5, 5)):
        assert out[yx] == 0.0
    assert np.all(np.isfinite(out))


def test_finalize_returns_fresh_copy_and_does_not_mutate_upstream():
    acc = DrizzleAccumulator((8, 8))
    acc._out_img[:] = np.arange(64, dtype=np.float32).reshape(8, 8)
    acc._out_wht[:] = 1.0
    img_before = acc._out_img.copy()
    wht_before = acc._out_wht.copy()
    out = acc.finalize("divide")
    assert out is not acc._out_img  # fresh copy, not a view
    out[0, 0] = 999.0
    assert np.array_equal(acc._out_img, img_before)
    assert np.array_equal(acc._out_wht, wht_before)


# --------------------------------------------------------------------------
# 7. real signed-Lanczos probe: negative WHT, no artificial huge science
# --------------------------------------------------------------------------


def test_lanczos2_signed_wht_no_huge_science():
    in_shape = (48, 48)
    out_shape = (64, 64)
    data = np.full(in_shape, 100.0, np.float32)
    yy, xx = np.indices(in_shape, dtype=np.float64)
    # sub-pixel-shifted partial coverage -> signed Lanczos lobes at the edge
    pixmap = np.dstack((xx + 8.5, yy + 8.3)).astype(np.float64)
    acc = DrizzleAccumulator(out_shape, kernel="lanczos2", pixfrac=1.0)
    acc.add(
        data,
        np.ones(in_shape, np.float32),
        pixmap,
        in_grid_mask=np.ones(in_shape, bool),
    )
    wht = acc.wht
    native = acc._out_img.copy()
    out = acc.finalize("divide")

    # the signed WHT is genuinely produced by the installed engine
    assert np.any(wht < 0.0), "expected negative native WHT from lanczos2 probe"
    # no artificial huge finite values (old formula produced ~-9.3e9 here)
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out))) <= float(np.max(np.abs(native))) + 1e-3
    # invariant: invalid (signed/non-positive) support -> 0 science
    assert support_integrity_violations(out, wht) == []


# --------------------------------------------------------------------------
# 8. square: new finalize vs legacy formula on positive support
# --------------------------------------------------------------------------


def test_square_new_vs_legacy_positive_support():
    shape = (16, 16)
    yy, xx = np.indices(shape, dtype=np.float64)
    pixmap = np.dstack((xx, yy)).astype(np.float64)

    acc = DrizzleAccumulator(shape)  # square kernel
    # frame 1: full, weight 1.0
    acc.add(np.full(shape, 10.0, np.float32), np.ones(shape, np.float32), pixmap)
    # frame 2: value 20.0, weight 0.5 (positive, non-trivial out_wht)
    w2 = np.full(shape, 0.5, np.float32)
    acc.add(np.full(shape, 20.0, np.float32), w2, pixmap)

    new = acc.finalize("divide")
    legacy = _legacy_finalize_divide(acc)
    metrics = _diff_metrics(new, legacy)

    # The only difference is float32 round-off of the extra multiply+divide in
    # the legacy formula (which reconstructed out_img * wht / wht).  It must be
    # far below any physical magnitude (~10..20 here).
    assert metrics["max_abs"] <= 1e-3, metrics
    assert metrics["rms"] <= 1e-4, metrics
    assert metrics["p99_9"] <= 1e-3, metrics
    # native science is materially unchanged on positive support
    assert np.allclose(new, acc._out_img, atol=1e-6)


# --------------------------------------------------------------------------
# 9. live preview consumes the same native-safe finalize("divide")
# --------------------------------------------------------------------------


def test_preview_uses_native_safe_finalize():
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    shape = (8, 8)
    accs = [DrizzleAccumulator(shape) for _ in range(3)]
    for acc in accs:
        acc._out_img[:] = 100.0
        acc._out_wht[:] = 1.0
    # signed negative WHT in one channel (must not produce huge preview values)
    accs[0]._out_wht[3, 3] = -0.5
    accs[0]._out_wht[4, 4] = -2.0

    qm = object.__new__(SeestarQueuedStacker)
    qm.drizzle_accumulators = accs
    qm.preview_downsample_factor = 1
    qm.current_stack_header = None
    qm._drizzle_frame_count = 1
    qm.files_in_queue = 1
    qm.stacked_batches_count = 0
    qm.total_batches_estimated = 0
    captured = []
    qm.preview_callback = lambda *args: captured.append(args)

    qm._update_preview_drizzle_accumulator()

    assert len(captured) == 1
    (legacy_norm, raw_linear), _hdr, _name, *_rest = captured[0]
    expected = np.stack([acc.finalize("divide") for acc in accs], axis=-1)
    # raw-linear preview IS the native-safe finalize("divide") HWC stack
    assert np.array_equal(raw_linear, expected.astype(np.float32))
    assert np.all(np.isfinite(raw_linear))
    assert float(np.max(np.abs(raw_linear))) <= 100.0  # no huge preview values


# --------------------------------------------------------------------------
# 10-11. queue_manager accumulator-boundary policy (effective vs requested)
# --------------------------------------------------------------------------


def _make_initialized_qm(tmp_path, kernel, pixfrac, threshold):
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    qm = SeestarQueuedStacker()
    qm.set_progress_callback(lambda *a, **k: None)
    qm.drizzle_active_session = True
    qm.is_mosaic_run = False
    qm.reference_wcs_object = make_wcs((32, 32))
    qm.drizzle_scale = 1.0
    qm.drizzle_kernel = kernel
    qm.drizzle_pixfrac = pixfrac
    qm.drizzle_wht_threshold = threshold
    qm.drizzle_output_wcs = make_wcs((32, 32))
    qm.drizzle_output_shape_hw = (32, 32)
    assert qm.initialize(str(tmp_path), (32, 32, 3)) is True
    return qm


def test_lanczos_effective_pixfrac_and_threshold_forced(tmp_path):
    qm = _make_initialized_qm(tmp_path, "lanczos2", 0.6, 0.3)
    for acc in qm.drizzle_accumulators:
        assert acc.kernel == "lanczos2"
        assert acc.pixfrac == 1.0  # upstream ignores pixfrac for Lanczos
    assert qm.drizzle_pixfrac_requested == 0.6
    assert qm.drizzle_wht_threshold_requested == 0.3
    assert qm.drizzle_wht_threshold_effective == 0.0


def test_square_effective_values_unchanged(tmp_path):
    qm = _make_initialized_qm(tmp_path, "square", 0.6, 0.3)
    for acc in qm.drizzle_accumulators:
        assert acc.kernel == "square"
        assert acc.pixfrac == 0.6
    assert qm.drizzle_pixfrac_requested == 0.6
    assert qm.drizzle_wht_threshold_requested == 0.3
    assert qm.drizzle_wht_threshold_effective == 0.3


# --------------------------------------------------------------------------
# 14 (unit). support-integrity gate detection
# --------------------------------------------------------------------------


def test_support_integrity_violations_detection():
    sci = np.zeros((4, 4, 3), dtype=np.float32)
    wht = np.ones((4, 4, 3), dtype=np.float32)
    # invalid support (wht <= epsilon) with nonzero science -> violation
    wht[0, 0, 0] = 0.0
    wht[1, 1, 1] = -0.5
    wht[2, 2, 2] = np.nan
    sci[0, 0, 0] = 500.0
    sci[1, 1, 1] = -3.0
    sci[2, 2, 2] = np.inf
    violations = support_integrity_violations(sci, wht)
    assert len(violations) == 3
    by_channel = {c: (n, m) for c, n, m in violations}
    assert by_channel[0][0] == 1 and by_channel[0][1] == 500.0
    assert by_channel[1][0] == 1 and by_channel[1][1] == 3.0
    assert by_channel[2][0] == 1
    # valid support with zero science -> no violation
    assert support_integrity_violations(np.zeros_like(sci), wht) == []


def test_lanczos_kernels_constant():
    assert LANCZOS_KERNELS == {"lanczos2", "lanczos3"}
