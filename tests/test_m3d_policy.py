"""M3-D-1 policy tests: standard vs incremental drizzle processing policy.

The M3 drizzle science is a single per-channel accumulator fed with the
ORIGINAL poses + transforms + weights.  ``drizzle_processing_policy`` only
changes resource/preview cadence (grouping + DISPLAY-ONLY previews), never the
science.  These tests assert:

* standard == incremental (identical SCI/WHT/WCS),
* group_size invariance (2 / 20 / 200),
* trailing partial group == full groups,
* no pose is double-counted or skipped (call count == N, weight linearity),
* previews are display-only and never mutate the accumulator state,
* accumulator memory footprint is independent of the number of poses.

Uses the same GUI-stub import pattern as ``test_worker_incremental_drizzle``
and the synthetic-frame / WCS helpers from ``test_drizzle_core``.  A lightweight
instance (``__new__``) is used for the accumulation-level tests so no worker
threads or process pools are spawned.
"""

import importlib
import math
import sys
import types
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

# Stub GUI modules to avoid Tk dependence during import (same as the worker
# incremental-drizzle harness).
if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = []
    settings_mod = types.ModuleType("seestar.gui.settings")

    class DummySettingsManager:
        pass

    settings_mod.SettingsManager = DummySettingsManager
    hist_mod = types.ModuleType("seestar.gui.histogram_widget")
    hist_mod.HistogramWidget = object
    gui_pkg.settings = settings_mod
    gui_pkg.histogram_widget = hist_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod
    sys.modules["seestar.gui.histogram_widget"] = hist_mod

qm = importlib.import_module("seestar.queuep.queue_manager")

from seestar.core.drizzle_core import (  # noqa: E402
    DrizzleAccumulator,
    build_output_grid,
    pixmap_from_alignment,
)


# --------------------------------------------------------------------------
# helpers (mirroring test_drizzle_core.py)
# --------------------------------------------------------------------------


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


def gauss2d(shape_hw, amp, sig, pos):
    h, w = shape_hw
    yy, xx = np.indices((h, w))
    return (amp * np.exp(-((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2) / (2.0 * sig ** 2))).astype(
        np.float32
    )


def identity_tf():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def translation_tf(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def rotation_tf(angle_deg, centre):
    a = math.radians(angle_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.array(centre, dtype=np.float64)
    t = c - r @ c
    return np.array([[r[0, 0], r[0, 1], t[0]], [r[1, 0], r[1, 1], t[1]]], dtype=np.float64)


def _synthetic_frames(shape=(32, 32), n=8):
    """Deterministic synthetic poses: (data_hwc float32, tf, weight_map).

    Each pose is a 3-channel Gaussian with a distinct sub-pixel shift and
    amplitude; the weight map has a spatially-varying low-weight band so WHT is
    non-uniform.
    """
    h, w = shape
    shifts = [
        (0.0, 0.0), (0.3, -0.7), (-0.5, 0.2), (0.8, 0.4),
        (-0.2, -0.4), (0.6, -0.1), (-0.7, 0.9), (0.1, 0.5),
    ]
    amps = [100.0, 80.0, 120.0, 60.0, 90.0, 110.0, 70.0, 130.0]
    centre = (w / 2.0, h / 2.0)
    frames = []
    for i in range(n):
        sx, sy = shifts[i % len(shifts)]
        amp = amps[i % len(amps)]
        if i % 7 == 6:
            tf = rotation_tf(8.0, centre)
        else:
            tf = translation_tf(sx, sy)
        base = gauss2d(shape, amp, 1.5, (centre[0] + sx, centre[1] + sy))
        data_hwc = np.stack([base * 1.0, base * 1.5, base * 0.7], axis=-1).astype(
            np.float32
        )
        weight = np.ones(shape, np.float32)
        weight[: h // 4, :] = 0.5
        frames.append((data_hwc, tf, weight))
    return frames


def _make_obj(shape, policy, group_size, preview_collector=None):
    """Lightweight accumulation harness (no worker loop, no process pools)."""
    obj = qm.SeestarQueuedStacker.__new__(qm.SeestarQueuedStacker)
    ref_wcs = make_wcs(shape)
    out_wcs, out_shape = build_output_grid(ref_wcs, shape, 1.0)
    obj.reference_wcs_object = ref_wcs
    obj.drizzle_output_wcs = out_wcs
    obj.drizzle_output_shape_hw = out_shape
    obj.drizzle_accumulators = [DrizzleAccumulator(out_shape) for _ in range(3)]
    obj.drizzle_processing_policy = policy
    obj.drizzle_group_size = max(1, int(group_size))
    obj._drizzle_frame_count = 0
    obj._drizzle_group_index = 0
    obj.preview_callback = preview_collector
    obj.current_stack_header = fits.Header()
    obj.files_in_queue = 0
    obj.stacked_batches_count = 0
    obj.total_batches_estimated = 0
    obj.preview_downsample_factor = 1
    return obj, out_wcs


def _run(policy, group_size, frames, preview_collector=None):
    shape = frames[0][0].shape[:2]
    obj, out_wcs = _make_obj(shape, policy, group_size, preview_collector)
    obj.files_in_queue = len(frames)
    for data_hwc, tf, wmap in frames:
        assert obj._add_frame_to_drizzle_accumulators(
            data_hwc, fits.Header(), tf, wmap, native_wcs=None
        )
        obj._drizzle_group_tick()
    obj._drizzle_flush_partial_group()
    sci = np.stack(
        [acc.finalize("divide") for acc in obj.drizzle_accumulators], axis=-1
    ).astype(np.float32)
    wht = np.stack([acc.wht for acc in obj.drizzle_accumulators], axis=-1).astype(
        np.float32
    )
    return sci, wht, out_wcs


def _assert_wcs_equal(a, b):
    assert np.allclose(a.wcs.cdelt, b.wcs.cdelt, atol=1e-9)
    assert np.allclose(a.wcs.crval, b.wcs.crval, atol=1e-9)
    assert np.allclose(a.wcs.crpix, b.wcs.crpix, atol=1e-9)
    assert list(a.wcs.ctype) == list(b.wcs.ctype)
    assert a.array_shape == b.array_shape


# --------------------------------------------------------------------------
# A) standard vs incremental -> identical SCI / WHT / WCS
# --------------------------------------------------------------------------


def test_standard_vs_incremental_identical():
    frames = _synthetic_frames(n=8)
    sci_std, wht_std, wcs_std = _run("standard", 50, frames)
    sci_inc, wht_inc, wcs_inc = _run("incremental", 2, frames)

    assert np.allclose(sci_std, sci_inc, atol=1e-6, rtol=1e-5)
    assert np.allclose(wht_std, wht_inc, atol=1e-6, rtol=1e-5)
    _assert_wcs_equal(wcs_std, wcs_inc)


# --------------------------------------------------------------------------
# B) incremental group_size 2 / 20 / 200 -> identical SCI / WHT
# --------------------------------------------------------------------------


def test_group_size_invariance():
    frames = _synthetic_frames(n=12)
    results = {}
    for gs in (2, 20, 200):
        sci, wht, _ = _run("incremental", gs, frames)
        results[gs] = (sci, wht)

    base_sci, base_wht = results[2]
    for gs in (20, 200):
        sci, wht = results[gs]
        assert np.allclose(sci, base_sci, atol=1e-6, rtol=1e-5)
        assert np.allclose(wht, base_wht, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------
# C) trailing partial group (N not a multiple of group_size) -> identical
# --------------------------------------------------------------------------


def test_last_partial_group_identical():
    frames = _synthetic_frames(n=7)
    sci_partial, wht_partial, _ = _run("incremental", 2, frames)  # 7 % 2 == 1
    sci_full, wht_full, _ = _run("incremental", 1, frames)  # 7 % 1 == 0

    assert np.allclose(sci_partial, sci_full, atol=1e-6, rtol=1e-5)
    assert np.allclose(wht_partial, wht_full, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------
# D) no pose double-counted or skipped; weight totals are linear
# --------------------------------------------------------------------------


def test_no_frame_double_or_skipped():
    frames = _synthetic_frames(n=8)
    shape = frames[0][0].shape[:2]
    obj, _ = _make_obj(shape, "incremental", 2, preview_collector=None)

    calls = {"n": 0}
    orig = obj._add_frame_to_drizzle_accumulators

    def counted(data, header, tf, wmap, native_wcs=None):
        calls["n"] += 1
        return orig(data, header, tf, wmap, native_wcs=native_wcs)

    obj._add_frame_to_drizzle_accumulators = counted
    for data_hwc, tf, wmap in frames:
        counted(data_hwc, fits.Header(), tf, wmap, native_wcs=None)
        obj._drizzle_group_tick()
    obj._drizzle_flush_partial_group()

    assert calls["n"] == len(frames)

    # Weight linearity: total accumulated weight must equal the sum of the
    # per-pose weights (each pose fed to a fresh accumulator).
    total_wht = float(sum(acc.wht.sum() for acc in obj.drizzle_accumulators))

    ref_wcs = obj.reference_wcs_object
    out_wcs = obj.drizzle_output_wcs
    per_pose_sum = 0.0
    for data_hwc, tf, wmap in frames:
        single = qm.SeestarQueuedStacker.__new__(qm.SeestarQueuedStacker)
        single.reference_wcs_object = ref_wcs
        single.drizzle_output_wcs = out_wcs
        single.drizzle_accumulators = [DrizzleAccumulator(shape) for _ in range(3)]
        assert single._add_frame_to_drizzle_accumulators(
            data_hwc, fits.Header(), tf, wmap, native_wcs=None
        )
        per_pose_sum += float(sum(acc.wht.sum() for acc in single.drizzle_accumulators))

    assert np.isclose(total_wht, per_pose_sum, rtol=1e-4)


# --------------------------------------------------------------------------
# E) previews are display-only and never mutate the accumulator state
# --------------------------------------------------------------------------


def test_preview_non_interfering():
    frames = _synthetic_frames(n=8)

    collected = []

    def collector(img, header, name, img_count, total, batch, total_batch):
        collected.append(np.asarray(img).copy())

    sci_with, wht_with, _ = _run("incremental", 2, frames, preview_collector=collector)
    sci_without, wht_without, _ = _run("incremental", 2, frames, preview_collector=None)

    assert np.allclose(sci_with, sci_without, atol=1e-6, rtol=1e-5)
    assert np.allclose(wht_with, wht_without, atol=1e-6, rtol=1e-5)
    # Previews were actually produced (4 full groups + 0 partial for n=8, gs=2)
    assert len(collected) > 0

    # Deep-compare the accumulator state before/after a preview call.
    obj, _ = _make_obj(frames[0][0].shape[:2], "incremental", 2, preview_collector=None)
    for data_hwc, tf, wmap in frames[:3]:
        obj._add_frame_to_drizzle_accumulators(data_hwc, fits.Header(), tf, wmap)
    obj._drizzle_frame_count = 3

    def snapshot():
        return [(a._out_img.copy(), a._out_wht.copy()) for a in obj.drizzle_accumulators]

    before = snapshot()
    obj.preview_callback = lambda *a, **k: None
    obj.current_stack_header = fits.Header()
    obj.files_in_queue = len(frames)
    obj.stacked_batches_count = 1
    obj.total_batches_estimated = 1
    obj.preview_downsample_factor = 1
    obj._update_preview_drizzle_accumulator()
    after = snapshot()

    for (bi, bw), (ai, aw) in zip(before, after):
        assert np.array_equal(bi, ai)
        assert np.array_equal(bw, aw)


# --------------------------------------------------------------------------
# F) accumulator memory footprint is independent of the number of poses
# --------------------------------------------------------------------------


def test_accumulator_memory_constant():
    shape = (16, 16)
    ref_wcs = make_wcs(shape)
    out_wcs, out_shape = build_output_grid(ref_wcs, shape, 1.0)
    tf = identity_tf()
    data = np.full((shape[0], shape[1], 3), 5.0, np.float32)
    wmap = np.ones(shape, np.float32)
    pixmap, mask = pixmap_from_alignment(shape, tf, ref_wcs, out_wcs)

    def nbytes_after(n):
        accs = [DrizzleAccumulator(out_shape) for _ in range(3)]
        for _ in range(n):
            for ch in range(3):
                accs[ch].add(data[..., ch], wmap, pixmap, in_grid_mask=mask)
        return sum(a._out_img.nbytes + a._out_wht.nbytes for a in accs)

    small = nbytes_after(10)
    large = nbytes_after(1000)
    assert small == large
    # and the footprint is exactly 3 channels * 2 arrays * H * W * 4 bytes
    assert small == 3 * 2 * shape[0] * shape[1] * np.dtype(np.float32).itemsize


# --------------------------------------------------------------------------
# G) _drizzle_group_tick: standard counts poses but never auto-previews;
#    incremental auto-previews per group. (M3-D-2 counter fix.)
# --------------------------------------------------------------------------


def _collector():
    collected = []

    def _collect(img, header, name, img_count, total, batch, total_batch):
        collected.append(np.asarray(img).copy())

    return collected, _collect


def test_tick_standard_counts_but_no_preview():
    frames = _synthetic_frames(n=6)
    shape = frames[0][0].shape[:2]
    collected, collector = _collector()
    obj, _ = _make_obj(shape, "standard", 2, preview_collector=collector)
    for data_hwc, tf, wmap in frames:
        assert obj._add_frame_to_drizzle_accumulators(
            data_hwc, fits.Header(), tf, wmap, native_wcs=None
        )
        obj._drizzle_group_tick()
    # Standard must still count every pose (manual preview reports exact count)...
    assert obj._drizzle_frame_count == 6
    # ...but must NOT trigger any automatic group preview.
    assert len(collected) == 0


def test_tick_incremental_previews_per_group():
    frames = _synthetic_frames(n=6)
    shape = frames[0][0].shape[:2]
    collected, collector = _collector()
    obj, _ = _make_obj(shape, "incremental", 2, preview_collector=collector)
    for data_hwc, tf, wmap in frames:
        assert obj._add_frame_to_drizzle_accumulators(
            data_hwc, fits.Header(), tf, wmap, native_wcs=None
        )
        obj._drizzle_group_tick()
    assert obj._drizzle_frame_count == 6
    # group_size=2, n=6 -> previews at frames 2, 4, 6.
    assert len(collected) == 3
