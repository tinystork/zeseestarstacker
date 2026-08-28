"""Synthetic integration tests for M3 Drizzle photometric integrity (DPIC-01).

These tests drive the real ``DrizzleAccumulator`` + ``pixmap_from_alignment``
path (and, where noted, the ``queue_manager`` boundary) with deterministic
synthetic RGB fields carrying different additive backgrounds, common stars and
a broad extended source.  They prove:

* the additive background match suppresses the spatial mean *step* produced by
  changing support, while the WHT/support change itself is preserved;
* a broad extended source spanning a coverage boundary keeps its structure
  (it is *not* flattened by the constant-offset correction);
* the correction composes with drizzle grouping (batch-size invariance);
* SCI/WHT coherence is intact.

The tf convention matches ``seestar.core.drizzle_core``: ``tf`` maps ORIGINAL
pixels to the REFERENCE grid.
"""

import math

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from seestar.core.drizzle_background import (
    BackgroundAnchor,
    apply_background_offsets,
    estimate_background_offsets,
    invert_affine_2x3,
)
from seestar.core.drizzle_core import (
    DrizzleAccumulator,
    build_output_grid,
    drizzle_stream,
    pixmap_from_alignment,
)
from seestar.queuep.queue_manager import SeestarQueuedStacker


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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


def identity_tf():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def translation_tf(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def gauss2d(shape_hw, amp, sig, pos):
    h, w = shape_hw
    yy, xx = np.indices((h, w))
    return (
        amp * np.exp(-((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2) / (2.0 * sig ** 2))
    ).astype(np.float32)


def build_rgb_frame(shape_hw, bg, tf, stars, source=None):
    """Build an RGB frame on its own pixel grid.

    ``bg`` is the per-channel additive background (scalar -> same for all
    channels).  ``stars`` is a list of ``(ref_x, ref_y, amp)`` in reference-grid
    coordinates, mapped back through ``inv(tf)`` onto the frame grid.  ``source``
    (optional) is a ``(H, W)`` pattern in reference coordinates, also mapped
    back onto the frame grid (broad extended source case).
    """
    h, w = shape_hw
    inv = invert_affine_2x3(np.asarray(tf, dtype=np.float64))
    img = np.full((h, w, 3), float(bg), dtype=np.float32)
    yy, xx = np.indices((h, w), dtype=np.float64)

    def add_pattern(pattern_ref):
        # pattern_ref is defined in REFERENCE coordinates; evaluate it at each
        # frame pixel's *forward*-mapped reference position tf @ (x, y).
        tf_arr = np.asarray(tf, dtype=np.float64)
        fx = tf_arr[0, 0] * xx + tf_arr[0, 1] * yy + tf_arr[0, 2]
        fy = tf_arr[1, 0] * xx + tf_arr[1, 1] * yy + tf_arr[1, 2]
        return pattern_ref(fx, fy)

    for sx, sy, amp in stars:
        fx = inv[0, 0] * sx + inv[0, 1] * sy + inv[0, 2]
        fy = inv[1, 0] * sx + inv[1, 1] * sy + inv[1, 2]
        for c in range(3):
            img[..., c] += gauss2d(shape_hw, amp, 1.5, (float(fx), float(fy)))

    if source is not None:
        for c in range(3):
            img[..., c] += add_pattern(source)

    return img.astype(np.float32)


def _drizzle_frames(shape, ref_wcs, out_wcs, frames, apply_correction, anchor=None):
    """Drizzle a list of ``(rgb, tf)`` frames into per-channel accumulators.

    When ``apply_correction`` is True, each frame is matched to ``anchor`` (the
    first frame by default) via ``estimate_background_offsets`` + constant
    subtract, then deposited.  Returns ``(sci_hwc, wht_hwc)``.
    """
    accs = [DrizzleAccumulator(out_wcs.array_shape) for _ in range(3)]
    if apply_correction and anchor is None:
        anchor = BackgroundAnchor(frames[0][0], reference_shape_hw=shape)
    for rgb, tf in frames:
        data = rgb
        if apply_correction:
            offsets, _ = estimate_background_offsets(
                data, np.ones(shape, np.float32), tf, anchor
            )
            data = apply_background_offsets(data, offsets)
        pixmap, in_grid = pixmap_from_alignment(shape, np.asarray(tf, np.float64), ref_wcs, out_wcs)
        for ch in range(3):
            accs[ch].add(data[..., ch], np.ones(shape, np.float32), pixmap, in_grid_mask=in_grid)
    sci = np.stack([a.finalize() for a in accs], axis=-1)
    wht = np.stack([a.wht for a in accs], axis=-1)
    return sci.astype(np.float32), wht.astype(np.float32)


# ---------------------------------------------------------------------------
# 1. translated partial-overlap + additive backgrounds + common stars
# ---------------------------------------------------------------------------


def test_translated_partial_overlap_mean_step_suppressed():
    shape = (64, 64)
    ref_wcs = make_wcs(shape)
    out_wcs, out_shape = build_output_grid(ref_wcs, shape, 1.0)

    # common stars (reference coordinates), placed at the top edge (y=2) so they
    # stay outside the measurement strips (y in [8, 56)).
    stars = [(16.0, 2.0, 2000.0), (32.0, 2.0, 2000.0), (48.0, 2.0, 2000.0)]

    # frame 0: full grid, bg 100 (the anchor); frame 1: x>=24 (bg 110);
    # frame 2: x<=39 (bg 90).  Coverage: left=2, middle=3, right=2 frames.
    f0 = (build_rgb_frame(shape, 100.0, identity_tf(), stars), identity_tf())
    f1 = (build_rgb_frame(shape, 110.0, translation_tf(24.0, 0.0), stars), translation_tf(24.0, 0.0))
    f2 = (build_rgb_frame(shape, 90.0, translation_tf(-24.0, 0.0), stars), translation_tf(-24.0, 0.0))
    frames = [f0, f1, f2]

    unc_sci, unc_wht = _drizzle_frames(shape, ref_wcs, out_wcs, frames, apply_correction=False)
    cor_sci, cor_wht = _drizzle_frames(shape, ref_wcs, out_wcs, frames, apply_correction=True)

    # region medians (luminance), away from stars: left strip / middle / right strip
    def region_median(sci, x0, x1):
        lum = 0.299 * sci[..., 0] + 0.587 * sci[..., 1] + 0.114 * sci[..., 2]
        return float(np.median(lum[8:56, x0:x1]))

    unc_left = region_median(unc_sci, 4, 16)
    unc_mid = region_median(unc_sci, 28, 36)
    unc_right = region_median(unc_sci, 48, 60)
    unc_step = max(unc_left, unc_mid, unc_right) - min(unc_left, unc_mid, unc_right)

    cor_left = region_median(cor_sci, 4, 16)
    cor_mid = region_median(cor_sci, 28, 36)
    cor_right = region_median(cor_sci, 48, 60)
    cor_step = max(cor_left, cor_mid, cor_right) - min(cor_left, cor_mid, cor_right)

    # uncorrected: the changing support creates a clear mean step (~10)
    assert unc_step > 2.0, f"expected an uncorrected mean step, got {unc_step:.3f}"
    # corrected: the step is strongly suppressed
    assert cor_step < 0.5, f"corrected step too large: {cor_step:.3f}"

    # WHT/support change remains (middle has HIGHER support than the sides)
    for wht in (unc_wht, cor_wht):
        w2d = np.mean(wht, axis=-1)
        side_support = float(np.mean(w2d[8:56, 4:16]))
        mid_support = float(np.mean(w2d[8:56, 28:36]))
        assert mid_support > side_support + 0.5


# ---------------------------------------------------------------------------
# 2. broad extended source spanning a coverage boundary
# ---------------------------------------------------------------------------


def test_extended_source_structure_preserved():
    shape = (64, 64)
    ref_wcs = make_wcs(shape)
    out_wcs, out_shape = build_output_grid(ref_wcs, shape, 1.0)

    bg = 100.0
    amp = 300.0
    sig = 10.0

    def broad_source(fx, fy):
        return amp * np.exp(-((fx - 32.0) ** 2 + (fy - 32.0) ** 2) / (2.0 * sig ** 2))

    f0 = (build_rgb_frame(shape, bg, identity_tf(), [], source=broad_source), identity_tf())
    # frame 1 covers only the right-hand ~2/3 (coverage boundary at x=20)
    f1 = (build_rgb_frame(shape, bg + 4.0, translation_tf(20.0, 0.0), [], source=broad_source),
          translation_tf(20.0, 0.0))

    # clean reference: the anchor frame alone (no partial overlap, no correction)
    clean_sci, _ = _drizzle_frames(shape, ref_wcs, out_wcs, [f0], apply_correction=False)
    cor_sci, _ = _drizzle_frames(shape, ref_wcs, out_wcs, [f0, f1], apply_correction=True)
    unc_sci, _ = _drizzle_frames(shape, ref_wcs, out_wcs, [f0, f1], apply_correction=False)

    lum_clean = 0.299 * clean_sci[..., 0] + 0.587 * clean_sci[..., 1] + 0.114 * clean_sci[..., 2]
    lum_cor = 0.299 * cor_sci[..., 0] + 0.587 * cor_sci[..., 1] + 0.114 * cor_sci[..., 2]
    lum_unc = 0.299 * unc_sci[..., 0] + 0.587 * unc_sci[..., 1] + 0.114 * unc_sci[..., 2]

    # 1) structure preserved: the corrected 2-frame output matches the clean
    #    single-frame reference (the broad source is not flattened or shifted).
    assert np.allclose(lum_cor, lum_clean, atol=1.5)

    # 2) not flattened: the source keeps its full amplitude.
    peak = float(np.max(lum_cor))
    far_bg = float(np.median(lum_cor[2:14, 2:14]))
    assert peak - far_bg > 0.8 * amp, f"source flattened: peak-bg={peak - far_bg:.1f}"

    # 3) the uncorrected output shows a background step across the coverage
    #    boundary (x=20): left (frame0-only) vs right (frame0+frame1).
    left_unc = float(np.median(lum_unc[8:56, 4:16]))
    right_unc = float(np.median(lum_unc[8:56, 44:60]))
    left_cor = float(np.median(lum_cor[8:56, 4:16]))
    right_cor = float(np.median(lum_cor[8:56, 44:60]))
    assert abs(right_unc - left_unc) > abs(right_cor - left_cor) + 1.0


# ---------------------------------------------------------------------------
# 3. group / batch-size invariance of the same ordered corrected frames
# ---------------------------------------------------------------------------


def test_group_size_invariance_with_correction():
    shape = (32, 32)
    ref_wcs = make_wcs(shape)
    out_wcs, out_shape = build_output_grid(ref_wcs, shape, 1.0)

    shifts = [(0.0, 0.0), (3.0, 0.0), (-3.0, 2.0), (0.0, 4.0)]
    bgs = [100.0, 104.0, 97.0, 101.0]
    stars = [(16.0, 16.0, 1500.0)]

    frames = [
        (build_rgb_frame(shape, bgs[i], translation_tf(*shifts[i]), stars), translation_tf(*shifts[i]))
        for i in range(4)
    ]
    anchor = BackgroundAnchor(frames[0][0], reference_shape_hw=shape)

    def gen():
        for rgb, tf in frames:
            offsets, _ = estimate_background_offsets(rgb, np.ones(shape, np.float32), tf, anchor)
            data = apply_background_offsets(rgb, offsets)
            yield (np.moveaxis(data, -1, 0), np.ones((3, *shape), np.float32), tf, 1.0, "counts")

    results = {}
    for gs in (1, 4):
        accs = [DrizzleAccumulator(out_shape) for _ in range(3)]
        sci, wht = drizzle_stream(accs, gen(), group_size=gs, reference_wcs=ref_wcs, output_wcs=out_wcs)
        results[gs] = (sci.copy(), wht.copy())

    s1, w1 = results[1]
    s4, w4 = results[4]
    assert np.allclose(s1, s4, atol=1e-6, rtol=1e-5)
    assert np.allclose(w1, w4, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. SCI/WHT coherence after correction
# ---------------------------------------------------------------------------


def test_sci_wht_coherence_after_correction():
    shape = (32, 32)
    ref_wcs = make_wcs(shape)
    out_wcs, out_shape = build_output_grid(ref_wcs, shape, 1.0)

    stars = [(16.0, 16.0, 1200.0)]
    frames = [
        (build_rgb_frame(shape, 100.0, identity_tf(), stars), identity_tf()),
        (build_rgb_frame(shape, 106.0, translation_tf(4.0, 0.0), stars), translation_tf(4.0, 0.0)),
    ]
    sci, wht = _drizzle_frames(shape, ref_wcs, out_wcs, frames, apply_correction=True)

    # In every channel, SCI must be the exposure-weighted mean (weighted flux /
    # weight): with unit weights and two corrected frames, a pixel covered by
    # both frames is exactly the mean of the two (background ~100, star present).
    # Coherence: wherever WHT == 2, the science equals the mean of the two
    # (corrected) inputs at that sky position; more simply, SCI is finite and
    # zero exactly where WHT == 0.
    zero_mask = np.all(wht == 0.0, axis=-1)
    assert np.all(np.all(sci[zero_mask] == 0.0, axis=-1))

    # interior (fully covered) region: background matched to ~100 per channel
    interior = np.all(wht >= 1.9, axis=-1)
    for ch in range(3):
        assert abs(float(np.median(sci[interior, ch])) - 100.0) < 1.0


# ---------------------------------------------------------------------------
# 5. queue_manager boundary: anchor capture + kernel/pixfrac wiring
# ---------------------------------------------------------------------------


def _make_fake_qm(shape, anchor=None):
    qm = object.__new__(SeestarQueuedStacker)
    qm.drizzle_active_session = True
    qm.is_mosaic_run = False
    qm.reference_wcs_object = make_wcs(shape)
    qm.drizzle_output_wcs, out_shape = build_output_grid(qm.reference_wcs_object, shape, 1.0)
    qm.drizzle_accumulators = [DrizzleAccumulator(out_shape) for _ in range(3)]
    qm.reference_shape = shape
    qm._registration_target_provenance_id = None
    qm._drizzle_bg_anchor = anchor
    qm._drizzle_bg_last_diag = None
    qm.update_progress = lambda *a, **k: None
    return qm


def _spy_drizzle_add(monkeypatch):
    """Install a per-channel deposition spy; return ``(captured, deposit)``."""
    captured = []
    orig_add = DrizzleAccumulator.add

    def spy_add(self, data, weight_map, pixmap, exptime=1.0, in_units="counts",
                in_grid_mask=None):
        captured.append(np.array(data, copy=True))
        return orig_add(self, data, weight_map, pixmap, exptime=exptime,
                        in_units=in_units, in_grid_mask=in_grid_mask)

    monkeypatch.setattr(DrizzleAccumulator, "add", spy_add)

    def deposit(qm, rgb, header, tf, weight):
        before = len(captured)
        qm._add_frame_to_drizzle_accumulators(rgb, header, tf, weight)
        # captured entries are per-channel 2-D arrays (3 channels per frame)
        return [float(np.median(c)) for c in captured[before:]]

    return captured, deposit


def test_qm_add_frame_applies_background_match(monkeypatch):
    shape = (32, 32)
    anchor = BackgroundAnchor(
        np.full((*shape, 3), 100.0, np.float32),
        reference_shape_hw=shape,
        provenance="reference:ref.fit",
    )
    qm = _make_fake_qm(shape, anchor=anchor)

    captured, deposit = _spy_drizzle_add(monkeypatch)
    header = fits.Header()
    header["EXPTIME"] = 1.0
    weight = np.ones(shape, np.float32)

    # frame A: bg 100 (the reference itself) -> ~zero correction
    med_a = deposit(
        qm, np.full((*shape, 3), 100.0, np.float32), header, identity_tf(),
        weight,
    )

    # frame B: bg 104 -> matched to 100 before deposition
    med_b = deposit(
        qm, np.full((*shape, 3), 104.0, np.float32), header, identity_tf(),
        weight,
    )

    # 3 channels x 2 frames captured (spy fired per channel)
    assert len(captured) == 6
    assert abs(med_a[0] - 100.0) < 0.5  # frame A ch0 (reference -> ~0 correction)
    assert abs(med_b[0] - 100.0) < 0.5  # frame B ch0 (corrected)


def test_qm_anchor_capture_from_reference_rescales_01_to_adu():
    shape = (32, 32)
    qm = _make_fake_qm(shape)
    # [0,1]-range reference (bg 100/65535) rescaled to ADU by the capture path.
    ref_01 = np.full((*shape, 3), 100.0 / 65535.0, np.float32)
    anchor = qm._capture_reference_drizzle_bg_anchor(ref_01, "frame_ref.fit")
    assert anchor is not None
    assert qm._drizzle_bg_anchor is anchor
    assert anchor.provenance == "reference:frame_ref.fit"
    # identity geometry
    assert np.allclose(anchor.tf, identity_tf(), atol=1e-12)
    # background rescaled ~100 ADU
    assert np.allclose(anchor.background, [100.0] * 3, atol=0.5)


def test_qm_first_frame_not_reference_anchor_stays_reference(monkeypatch):
    shape = (32, 32)
    qm = _make_fake_qm(shape)
    ref_01 = np.full((*shape, 3), 100.0 / 65535.0, np.float32)
    qm._capture_reference_drizzle_bg_anchor(ref_01, "frame_ref.fit")

    captured, deposit = _spy_drizzle_add(monkeypatch)
    header = fits.Header()
    header["EXPTIME"] = 1.0
    weight = np.ones(shape, np.float32)

    # first accepted frame is NOT the reference (bg 110).
    med = deposit(
        qm, np.full((*shape, 3), 110.0, np.float32), header, identity_tf(),
        weight,
    )
    # anchor provenance/data remain the selected reference, not the first frame.
    assert qm._drizzle_bg_anchor.provenance == "reference:frame_ref.fit"
    assert np.allclose(qm._drizzle_bg_anchor.background, [100.0] * 3, atol=0.5)
    # the first (non-reference) frame was corrected toward the reference.
    assert abs(med[0] - 100.0) < 0.5


def test_qm_reference_yields_zero_correction(monkeypatch):
    shape = (32, 32)
    anchor = BackgroundAnchor(
        np.full((*shape, 3), 100.0, np.float32),
        reference_shape_hw=shape,
        provenance="reference:ref.fit",
    )
    qm = _make_fake_qm(shape, anchor=anchor)
    captured, deposit = _spy_drizzle_add(monkeypatch)
    header = fits.Header()
    header["EXPTIME"] = 1.0
    weight = np.ones(shape, np.float32)
    med = deposit(
        qm, np.full((*shape, 3), 100.0, np.float32), header, identity_tf(),
        weight,
    )
    assert abs(med[0] - 100.0) < 0.5
    assert qm._drizzle_bg_last_diag["reason"] == "accepted"
    assert np.allclose(qm._drizzle_bg_last_diag["offsets"], 0.0, atol=0.1)


def test_qm_arrival_order_invariance(monkeypatch):
    shape = (32, 32)

    def run(order):
        anchor = BackgroundAnchor(
            np.full((*shape, 3), 100.0, np.float32),
            reference_shape_hw=shape,
            provenance="reference:ref.fit",
        )
        qm = _make_fake_qm(shape, anchor=anchor)
        captured, deposit = _spy_drizzle_add(monkeypatch)
        header = fits.Header()
        header["EXPTIME"] = 1.0
        weight = np.ones(shape, np.float32)
        out = {}
        for bg in order:
            med = deposit(
                qm, np.full((*shape, 3), bg, np.float32), header, identity_tf(),
                weight,
            )
            out[bg] = med[0]
        return out

    r1 = run([104.0, 97.0, 101.0])
    r2 = run([101.0, 104.0, 97.0])
    # each frame's correction is a pure function of the immutable anchor + frame
    for bg in (104.0, 97.0, 101.0):
        assert abs(r1[bg] - r2[bg]) < 1e-4
    # and both are corrected toward the reference background (~100)
    for bg in (104.0, 97.0, 101.0):
        assert abs(r1[bg] - 100.0) < 0.5


def test_qm_no_anchor_neutral_fallback(monkeypatch):
    shape = (32, 32)
    qm = _make_fake_qm(shape)  # no anchor
    captured, deposit = _spy_drizzle_add(monkeypatch)
    header = fits.Header()
    header["EXPTIME"] = 1.0
    weight = np.ones(shape, np.float32)
    med = deposit(
        qm, np.full((*shape, 3), 104.0, np.float32), header, identity_tf(),
        weight,
    )
    # neutral: deposited verbatim (no correction), no arrival-dependent anchor.
    assert qm._drizzle_bg_anchor is None
    assert qm._drizzle_bg_last_diag["reason"] == "no_anchor"
    assert abs(med[0] - 104.0) < 0.5


def test_qm_initialize_wires_kernel_pixfrac(tmp_path):
    qm = SeestarQueuedStacker()
    qm.set_progress_callback(lambda *a, **k: None)
    qm.drizzle_active_session = True
    qm.is_mosaic_run = False
    qm.reference_wcs_object = make_wcs((32, 32))
    qm.drizzle_scale = 1.0
    qm.drizzle_kernel = "gaussian"
    qm.drizzle_pixfrac = 0.8
    qm.drizzle_output_wcs = make_wcs((32, 32))
    qm.drizzle_output_shape_hw = (32, 32)

    ok = qm.initialize(str(tmp_path), (32, 32, 3))
    assert ok is True
    for acc in qm.drizzle_accumulators:
        assert acc.kernel == "gaussian"
        assert acc.pixfrac == 0.8


def test_qm_initialize_invalid_kernel_pixfrac_fallbacks(tmp_path):
    qm = SeestarQueuedStacker()
    qm.set_progress_callback(lambda *a, **k: None)
    qm.drizzle_active_session = True
    qm.is_mosaic_run = False
    qm.reference_wcs_object = make_wcs((32, 32))
    qm.drizzle_scale = 1.0
    qm.drizzle_kernel = "not-a-kernel"
    qm.drizzle_pixfrac = 9.0
    qm.drizzle_output_wcs = make_wcs((32, 32))
    qm.drizzle_output_shape_hw = (32, 32)

    ok = qm.initialize(str(tmp_path), (32, 32, 3))
    assert ok is True
    for acc in qm.drizzle_accumulators:
        assert acc.kernel == "square"
        assert acc.pixfrac == 1.0
