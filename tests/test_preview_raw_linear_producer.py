"""ZSSS-OTPUX-PREVIEW-CORE-01 — backend raw-linear payload seam tests.

Verifies the Option-A backend preview producers in
``seestar/queuep/queue_manager.py``:

* ``_update_preview_sum_w`` (classic SUM/W) and
  ``_update_preview_drizzle_accumulator`` (M3 Drizzle) both send a
  ``(legacy_normalized, raw_linear)`` tuple;
* the first element is pixel-identical to the previous single-array callback
  (backward compatibility);
* the second element is the raw-linear float source (classic: SUM/W divide
  after display-only masks, before min/max; Drizzle: ``finalize("divide")``
  HWC before the 1%/99% percentile stretch), at the *same* final geometry
  (including the default 2x downsample);
* neither producer mutates the SUM/WHT memmaps, the Drizzle accumulators, or
  the input arrays (scientific-separation witness).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seestar.queuep.queue_manager import SeestarQueuedStacker  # noqa: E402
from seestar.core.drizzle_core import DrizzleAccumulator  # noqa: E402


def _classic_stack():
    obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    obj.update_progress = lambda *a, **k: None
    obj.preview_callback = None
    obj.current_stack_header = fits.Header()
    obj.images_in_cumulative_stack = 3
    obj.files_in_queue = 10
    obj.stacked_batches_count = 1
    obj.total_batches_estimated = 2
    return obj


def _drizzle_stack():
    obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    obj.update_progress = lambda *a, **k: None
    obj.preview_callback = None
    obj.current_stack_header = fits.Header()
    obj._drizzle_frame_count = 2
    obj.files_in_queue = 5
    obj.stacked_batches_count = 1
    obj.total_batches_estimated = 2
    return obj


# ---------------------------------------------------------------------------
# Classic SUM/W producer
# ---------------------------------------------------------------------------

def test_classic_producer_tuple_and_backward_compat():
    obj = _classic_stack()
    obj.preview_downsample_factor = 1  # no downsample for exact value checks
    H, W = 8, 8
    rng = np.random.default_rng(20)
    avg = rng.uniform(0.0, 100.0, size=(H, W, 3)).astype(np.float32)
    obj.cumulative_sum_memmap = avg.astype(np.float32)  # WHT == 1 -> SUM/W = avg
    obj.cumulative_wht_memmap = np.ones((H, W), dtype=np.float32)

    collected = []
    obj.preview_callback = lambda *a: collected.append(a)
    obj._update_preview_sum_w()

    assert len(collected) == 1
    payload = collected[0][0]
    assert isinstance(payload, tuple) and len(payload) == 2
    legacy, raw = payload

    # First element == previous normalized (min/max + clip) single-array output.
    avg64 = avg.astype(np.float64)  # the producer normalizes in float64
    mn = float(np.nanmin(avg64))
    mx = float(np.nanmax(avg64))
    expected_legacy = np.clip((avg64 - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)
    assert np.array_equal(legacy, expected_legacy)

    # Second element == raw-linear SUM/W divide (no masks here) == avg.
    assert np.array_equal(raw, avg)
    assert legacy.shape == raw.shape == (H, W, 3)


def test_classic_producer_default_2x_geometry():
    obj = _classic_stack()  # preview_downsample_factor not set -> default 2
    H, W = 40, 60
    avg = np.linspace(0.0, 1.0, H * W * 3, dtype=np.float32).reshape(H, W, 3)
    obj.cumulative_sum_memmap = avg.astype(np.float32)
    obj.cumulative_wht_memmap = np.ones((H, W), dtype=np.float32)

    collected = []
    obj.preview_callback = lambda *a: collected.append(a)
    obj._update_preview_sum_w()

    legacy, raw = collected[0][0]
    assert legacy.shape == (H // 2, W // 2, 3)
    assert raw.shape == (H // 2, W // 2, 3)
    assert legacy.shape == raw.shape


# ---------------------------------------------------------------------------
# Drizzle producer
# ---------------------------------------------------------------------------

def _fill_drizzle(obj, shape):
    obj.drizzle_accumulators = [DrizzleAccumulator(shape) for _ in range(3)]
    for i, acc in enumerate(obj.drizzle_accumulators):
        acc._out_img[:] = float(i + 1) * 10.0
        acc._out_wht[:] = 2.0


def test_drizzle_producer_tuple_and_backward_compat():
    obj = _drizzle_stack()
    obj.preview_downsample_factor = 1
    shape = (8, 8)
    _fill_drizzle(obj, shape)

    collected = []
    obj.preview_callback = lambda *a: collected.append(a)
    obj._update_preview_drizzle_accumulator()

    assert len(collected) == 1
    payload = collected[0][0]
    assert isinstance(payload, tuple) and len(payload) == 2
    legacy, raw = payload

    # Second element == finalize("divide") HWC BEFORE the percentile stretch.
    channels = [acc.finalize("divide").astype(np.float32) for acc in obj.drizzle_accumulators]
    expected_raw = np.stack(channels, axis=-1)
    assert np.array_equal(raw, expected_raw)

    # First element == previous 1%/99% percentile-stretched output.
    with np.errstate(all="ignore"):
        lo, hi = np.nanpercentile(expected_raw, [1.0, 99.0])
    expected_legacy = np.clip((expected_raw - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    assert np.array_equal(legacy, expected_legacy)
    assert legacy.shape == raw.shape == (8, 8, 3)


def test_drizzle_producer_default_2x_geometry():
    obj = _drizzle_stack()  # preview_downsample_factor not set -> default 2
    shape = (40, 60)
    _fill_drizzle(obj, shape)

    collected = []
    obj.preview_callback = lambda *a: collected.append(a)
    obj._update_preview_drizzle_accumulator()

    legacy, raw = collected[0][0]
    assert legacy.shape == (shape[0] // 2, shape[1] // 2, 3)
    assert raw.shape == (shape[0] // 2, shape[1] // 2, 3)
    assert legacy.shape == raw.shape


# ---------------------------------------------------------------------------
# Scientific-separation witness: producers never mutate science state/inputs
# ---------------------------------------------------------------------------

def test_producers_do_not_mutate_accumulators_or_inputs():
    # Classic: SUM/WHT inputs unchanged.
    obj = _classic_stack()
    obj.preview_downsample_factor = 1
    H, W = 8, 8
    rng = np.random.default_rng(30)
    sum_mm = rng.uniform(0.0, 10.0, size=(H, W, 3)).astype(np.float32)
    sum_before = sum_mm.copy()
    wht_mm = np.full((H, W), 2.0, dtype=np.float32)
    wht_before = wht_mm.copy()
    obj.cumulative_sum_memmap = sum_mm
    obj.cumulative_wht_memmap = wht_mm
    collected = []
    obj.preview_callback = lambda *a: collected.append(a)
    obj._update_preview_sum_w()
    assert np.array_equal(sum_mm, sum_before)
    assert np.array_equal(wht_mm, wht_before)

    # Drizzle: accumulator state unchanged.
    obj2 = _drizzle_stack()
    obj2.preview_downsample_factor = 1
    shape = (8, 8)
    _fill_drizzle(obj2, shape)
    img_before = [a._out_img.copy() for a in obj2.drizzle_accumulators]
    wht_before = [a._out_wht.copy() for a in obj2.drizzle_accumulators]
    collected2 = []
    obj2.preview_callback = lambda *a: collected2.append(a)
    obj2._update_preview_drizzle_accumulator()
    for acc, bi, bw in zip(obj2.drizzle_accumulators, img_before, wht_before):
        assert np.array_equal(acc._out_img, bi)
        assert np.array_equal(acc._out_wht, bw)
