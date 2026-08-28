"""R2.1 — shared RGB photometric domain ("WB basique") tests for DPIC-01.

The R1 review found that the Drizzle background anchor was rescaled into ADU
without the per-frame "WB basique" R/B gains that ``_process_file`` applies,
so the anchor and the deposited frames differed *multiplicatively* in R/B.
This file proves the fix: the exact gains are extracted into a shared helper
(``apply_wb_basique``), ``_process_file`` and ``_capture_reference_drizzle_bg_anchor``
both use it, and the resulting anchor/frame photometric domains match.
"""

import numpy as np
import pytest

from seestar.core.drizzle_background import (
    BackgroundAnchor,
    apply_background_offsets,
    apply_wb_basique,
    estimate_background_offsets,
    rescale_01_to_adu,
)
from seestar.queuep.queue_manager import SeestarQueuedStacker


def _identity_tf():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _bare_qm():
    qm = object.__new__(SeestarQueuedStacker)
    qm._drizzle_bg_anchor = None
    return qm


# ---------------------------------------------------------------------------
# 1. exact gains for asymmetric RGB medians (source helper + anchor capture)
# ---------------------------------------------------------------------------


def test_wb_basique_exact_gains_asymmetric():
    # medians R=1, G=2, B=4 -> gain_r = clip(2/1)=2.0, gain_b = clip(2/4)=0.5.
    img = np.stack(
        [
            np.full((16, 16), 1.0, np.float32),
            np.full((16, 16), 2.0, np.float32),
            np.full((16, 16), 4.0, np.float32),
        ],
        -1,
    )
    original = img.copy()
    out, info = apply_wb_basique(img)

    assert info["applied"] is True
    assert info["gain_r"] == pytest.approx(2.0, rel=1e-6)
    assert info["gain_b"] == pytest.approx(0.5, rel=1e-6)
    # R doubled, B halved, G untouched -> all channels equal 2.0
    assert np.allclose(out[..., 0], 2.0, atol=1e-5)
    assert np.allclose(out[..., 1], 2.0, atol=1e-5)
    assert np.allclose(out[..., 2], 2.0, atol=1e-5)
    # input never mutated
    assert np.array_equal(img, original)


def test_anchor_capture_applies_same_wb_gains():
    # The reference returned by _get_reference_image is NOT WB'd; the anchor
    # capture must apply the same gains as the source path before ADU rescale.
    qm = _bare_qm()
    # R median 0.25, G 0.5, B 1.0 (in [0,1]) -> WB pulls R and B toward G (0.5).
    ref_01 = np.stack(
        [
            np.full((16, 16), 0.25, np.float32),
            np.full((16, 16), 0.5, np.float32),
            np.full((16, 16), 1.0, np.float32),
        ],
        -1,
    )
    anchor = qm._capture_reference_drizzle_bg_anchor(ref_01, "frame_ref.fit")

    # After WB all three channels have median 0.5, so after *65535 the anchor
    # backgrounds are equal (same RGB domain), not the raw 0.25/0.5/1.0 ratios.
    assert np.allclose(anchor.background, anchor.background[0], atol=1e-2)
    assert np.allclose(anchor.background, 0.5 * 65535.0, atol=1.0)


# ---------------------------------------------------------------------------
# 2. source path vs captured anchor agree (tight tolerance)
# ---------------------------------------------------------------------------


def test_reference_source_path_agrees_with_captured_anchor():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.0, 0.5, (32, 32, 3)).astype(np.float32)
    raw[..., 0] *= 0.6
    raw[..., 2] *= 1.4

    # source-path photometric preparation (WB then ADU rescale)
    wb, _ = apply_wb_basique(raw)
    source_adu = rescale_01_to_adu(wb)

    # anchor-capture path (must be bit-identical to the source path)
    qm = _bare_qm()
    anchor = qm._capture_reference_drizzle_bg_anchor(raw, "frame_ref.fit")

    assert anchor._data.dtype == np.float32
    assert np.array_equal(anchor._data, source_adu)


# ---------------------------------------------------------------------------
# 3. coloured diffuse structure + unequal medians: offsets recovered, colour
#    preserved (this test fails against the R1 non-WB anchor)
# ---------------------------------------------------------------------------


def test_colored_diffuse_structure_offsets_recovered():
    shape = (64, 64)
    yy, xx = np.indices(shape)
    g = np.exp(-((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / (2 * 8.0 ** 2)).astype(np.float32)
    base = np.array([0.2, 0.4, 0.8], np.float32)   # unequal per-channel medians
    amp = np.array([0.3, 0.6, 0.15], np.float32)    # coloured broad Gaussian
    raw = np.empty((*shape, 3), np.float32)
    for c in range(3):
        raw[..., c] = base[c] + amp[c] * g

    # R2: the anchor is captured through the same WB+ADU domain as the frames.
    qm = _bare_qm()
    anchor = qm._capture_reference_drizzle_bg_anchor(raw, "ref.fit")

    # A second capture of the same sky: source-prepared (WB then ADU) plus an
    # injected additive per-channel sky offset.
    wb_frame, _ = apply_wb_basique(raw)
    frame_adu = rescale_01_to_adu(wb_frame)
    injected = np.array([12.0, -7.0, 25.0], np.float64)
    frame = (frame_adu + injected.reshape((1, 1, 3)).astype(np.float32)).astype(np.float32)

    offsets, diag = estimate_background_offsets(
        frame, np.ones(shape, np.float32), _identity_tf(), anchor
    )
    assert diag["reason"] == "accepted"
    # the recovered offsets are exactly the injected additive RGB offsets
    assert np.allclose(offsets, injected, atol=0.5)

    # the corrected frame is in the same RGB domain as the anchor and the
    # coloured extended structure is preserved (not flattened to the median)
    corrected = apply_background_offsets(frame, offsets)
    assert corrected.dtype == np.float32
    for c in range(3):
        band = corrected[..., c]
        contrast = float(band.max() - np.median(band))
        assert contrast > 0.5 * (amp[c] * 65535.0)

    # Counterfactual R1 behaviour: an anchor WITHOUT the WB gains (only ADU
    # rescale) sits in a different R/B domain, so the estimator cannot recover
    # the injected offsets (the coloured structure leaks into the delta).
    anchor_r1 = BackgroundAnchor(rescale_01_to_adu(raw), reference_shape_hw=shape)
    off_r1, _ = estimate_background_offsets(
        frame, np.ones(shape, np.float32), _identity_tf(), anchor_r1
    )
    assert not np.allclose(off_r1, injected, atol=0.5)


# ---------------------------------------------------------------------------
# 4. Classic preparation/output witness unchanged (bit-identical to inline)
# ---------------------------------------------------------------------------


def test_wb_basique_matches_inline_classic_formula():
    rng = np.random.default_rng(42)
    img = rng.uniform(0.0, 1.0, (24, 24, 3)).astype(np.float32)
    img[..., 0] *= 0.6
    img[..., 2] *= 1.4

    # baseline-equivalent inline formula (the exact legacy block, on a copy)
    ref = img.copy()
    r_ch, g_ch, b_ch = ref[..., 0], ref[..., 1], ref[..., 2]
    med_r, med_g, med_b = np.median(r_ch), np.median(g_ch), np.median(b_ch)
    if med_g > 1e-6:
        gain_r = np.clip(med_g / max(med_r, 1e-6), 0.5, 2.0)
        gain_b = np.clip(med_g / max(med_b, 1e-6), 0.5, 2.0)
        ref[..., 0] *= gain_r
        ref[..., 2] *= gain_b

    original = img.copy()
    out, info = apply_wb_basique(img)

    # the helper replacement is numerically (bit-for-bit) identical to the
    # legacy inline block, so the Classic output is unchanged.
    assert np.array_equal(out, ref)
    # and the input is never mutated
    assert np.array_equal(img, original)
