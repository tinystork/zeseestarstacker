"""Pipeline-order seam test: CFA correction runs BEFORE debayer and the RGB
hot-pixel pass is skipped after a successful CFA correction.

Mission ``ZSSS-HOT-PIXEL-SMALL-N-ROBUSTNESS-20260923``.

Uses the standalone ``seestar.core.geometry_reference.reference_quality_metric``
as the pipeline seam (it embodies the exact same CFA-before-debayer ordering as
the queue_manager main path and the alignment reference paths) and monkeypatches
the three helpers to prove call ordering — no E2E FITS run.
"""

from __future__ import annotations

import numpy as np
import pytest

import seestar.core.geometry_reference as gr
import seestar.core.hot_pixels as hp
import seestar.core.image_processing as ip


def _bayer_image(hot=False):
    img = np.full((16, 16), 100.0, dtype=np.float32)
    if hot:
        img[3, 3] = 63000.0
    return img


def _rgb_image():
    # Non-constant RGB so the reference-quality variance gate (std >= 0.0005)
    # does not early-return before the hot-pixel ordering is exercised.
    y, x = np.mgrid[0:16, 0:16]
    plane = (x * 10.0).astype(np.float32)
    return np.stack([plane] * 3, axis=-1)


def test_cfa_correction_runs_before_debayer_and_skips_rgb(monkeypatch):
    calls = []

    monkeypatch.setattr(
        ip,
        "load_and_validate_fits",
        lambda path: (_bayer_image(hot=True), {"BAYERPAT": "RGGB"}),
    )

    def fake_cfa(image, pattern, threshold=3.0, neighborhood_size=5):
        calls.append("cfa")
        return image, {"enabled": True, "pattern": pattern, "candidates": 1, "corrected": 1}

    def fake_debayer(image, pattern="GRBG"):
        calls.append("debayer")
        return _rgb_image()

    def fake_rgb(image, threshold=3.0, neighborhood_size=5):
        calls.append("rgb")
        return image

    monkeypatch.setattr(hp, "detect_and_correct_hot_pixels_cfa", fake_cfa)
    monkeypatch.setattr(ip, "debayer_image", fake_debayer)
    monkeypatch.setattr(hp, "detect_and_correct_hot_pixels", fake_rgb)

    gr.reference_quality_metric(
        "/fake/raw.fits", bayer_pattern="RGGB", correct_hot_pixels=True,
    )

    # CFA was invoked before debayer, and the RGB pass was skipped entirely.
    assert calls == ["cfa", "debayer"]


def test_no_valid_pattern_skips_cfa_and_keeps_rgb(monkeypatch):
    calls = []

    monkeypatch.setattr(
        ip,
        "load_and_validate_fits",
        lambda path: (_bayer_image(hot=True), {"BAYERPAT": "MONO"}),
    )

    def fake_cfa(image, pattern, threshold=3.0, neighborhood_size=5):
        calls.append("cfa")
        return image, {"enabled": True, "pattern": pattern, "candidates": 0, "corrected": 0}

    def fake_debayer(image, pattern="GRBG"):
        calls.append("debayer")
        return _rgb_image()

    def fake_rgb(image, threshold=3.0, neighborhood_size=5):
        calls.append("rgb")
        return image

    monkeypatch.setattr(hp, "detect_and_correct_hot_pixels_cfa", fake_cfa)
    monkeypatch.setattr(ip, "debayer_image", fake_debayer)
    monkeypatch.setattr(hp, "detect_and_correct_hot_pixels", fake_rgb)

    gr.reference_quality_metric(
        "/fake/mono.fits", bayer_pattern="GRBG", correct_hot_pixels=True,
    )

    # No reliable Bayer pattern: no CFA correction, no debayer, RGB pass kept.
    assert calls == ["rgb"]


def test_3d_rgb_input_skips_cfa_and_keeps_rgb(monkeypatch):
    calls = []

    monkeypatch.setattr(
        ip,
        "load_and_validate_fits",
        lambda path: (_rgb_image(), {"BAYERPAT": "RGGB"}),
    )

    def fake_cfa(image, pattern, threshold=3.0, neighborhood_size=5):
        calls.append("cfa")
        return image, {"enabled": True, "pattern": pattern, "candidates": 0, "corrected": 0}

    def fake_debayer(image, pattern="GRBG"):
        calls.append("debayer")
        return image

    def fake_rgb(image, threshold=3.0, neighborhood_size=5):
        calls.append("rgb")
        return image

    monkeypatch.setattr(hp, "detect_and_correct_hot_pixels_cfa", fake_cfa)
    monkeypatch.setattr(ip, "debayer_image", fake_debayer)
    monkeypatch.setattr(hp, "detect_and_correct_hot_pixels", fake_rgb)

    gr.reference_quality_metric(
        "/fake/rgb.fits", bayer_pattern="RGGB", correct_hot_pixels=True,
    )

    # 3D RGB input: no CFA correction, no debayer, RGB pass kept.
    assert calls == ["rgb"]
