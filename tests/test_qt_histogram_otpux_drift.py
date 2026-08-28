"""ZSSS-PREVIEW-DRIFT-HIST — histogram regression witness for anchor drift.

Focused offscreen Qt test that proves, against the *authoritative* live
histogram model (``MainWindow._histogram_model`` / ``right_histogram_view.model``),
that a legitimate 2x-3x photometric drift no longer collapses the whole
population into the top bin (``x = 1``) with ``median``/``mean`` == 1.0.

This is the regression witness for ZSSS-PREVIEW-DRIFT-01: the frozen
p0.5/p99.5 anchors used to hard-clip every pixel of a brighter successive
frame to ``1.0``, which the float histogram then reported as a single spike in
the last of its 512 bins.  The fix (a display-only hysteretic monotonic anchor
widening) keeps the histogram in-domain, so this test asserts the top-bin
fraction stays small and the per-channel ``median``/``mean`` stay well inside
``(0, 1)``.

This witness drives the **authoritative RGB channels directly** (not the mono
``L`` reduction): it feeds a deterministic channels-last HWC Option-A sequence
with three distinct, healthy channel distributions and asserts the applied
model reports ``R``/``G``/``B``.  The display mapping (anchors → ``[0, 1]``
map → float histogram) is *shared* by mono and RGB, so exercising the RGB
channels proves the exact symptom the real drift shows (R/G/B medians/means
collapsing toward ``1``) without a separate mono alias.

The test deliberately imports only stable public seams (``MainWindow``,
``BackendPreviewPayload``, ``create_application``): on the pre-fix baseline it
fails *behaviourally* (top-bin collapse / median == 1.0), not via ImportError.

No real stacking, no engine, no Tk.  ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import BackendPreviewPayload, MainWindow, create_application


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _pump_until(predicate, timeout_ms: int = 5000) -> bool:
    """Pump the Qt event loop until ``predicate`` is true (or time out)."""
    app = QApplication.instance()
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


def _wait_histogram(win: MainWindow, timeout_ms: int = 5000) -> bool:
    """Wait until the applied histogram model matches the current WB-only revision."""
    return _pump_until(
        lambda: win._histogram_model is not None
        and win._histogram_model_revision == win._wb_only_revision,
        timeout_ms,
    )


def _legacy_normalize(arr: np.ndarray) -> np.ndarray:
    """A deliberately misleading legacy-normalized copy (min/max -> [0, 1])."""
    arr64 = arr.astype(np.float64)
    mn = float(np.nanmin(arr64))
    mx = float(np.nanmax(arr64))
    return np.clip((arr64 - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)


def _option_a(win: MainWindow, raw: np.ndarray, name: str) -> None:
    """Feed an Option-A payload and wait until its histogram model is applied."""
    win._on_preview(
        BackendPreviewPayload(data=(_legacy_normalize(raw), raw), stack_name=name)
    )
    assert win._pristine_float is not None
    assert _wait_histogram(win)


def _top_bin_fraction(model, channel: str) -> float:
    """Fraction of the whole population that landed in the last (x=1) bin."""
    counts = model["counts"][channel]
    total = int(counts.sum())
    if total == 0:
        return 0.0
    return float(int(counts[-1]) / total)


def _rgb_frame(rng: np.random.Generator) -> np.ndarray:
    """Deterministic HWC RGB frame with three distinct healthy channel bands.

    R/G/B are independent ``uniform`` populations with distinct, well-separated
    medians (B > R > G) so the authoritative histogram channels are clearly
    distinguishable and each channel's median maps to a distinct, in-domain
    value.
    """
    r = rng.uniform(110.0, 210.0, size=(400, 400))
    g = rng.uniform(90.0, 190.0, size=(400, 400))
    b = rng.uniform(130.0, 230.0, size=(400, 400))
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def test_successive_drift_does_not_collapse_histogram_top_bin(qapp):
    """2x / 3x successive drift must not spike the RGB histogram top bin.

    Baseline (frozen anchors) collapses the entire brighter frame to ``1.0``:
    the authoritative histogram model reports ``median == mean == 1.0`` and a
    top-bin fraction of ``1.0`` on the R/G/B channels.  The fixed behaviour
    keeps the histogram in-domain (top-bin fraction << 1, ``median``/``mean``
    well inside ``(0, 1)``) on every channel.
    """
    win = MainWindow()
    try:
        rng = np.random.default_rng(20260828)
        frame1 = _rgb_frame(rng)

        _option_a(win, frame1, "f1")
        model1 = win._histogram_model
        assert model1 is not None
        assert model1 is win.right_histogram_view.model  # authoritative model
        assert model1["channels"] == ["R", "G", "B"]
        for ch in ("R", "G", "B"):
            assert model1["counts"][ch].shape == (512,)
            med = float(model1["stats"][ch]["median"])
            # A healthy first frame: median well inside the domain, not collapsed.
            assert 0.0 < med < 1.0
        # Three *distinct* channel distributions (B > R > G medians).
        med = {ch: float(model1["stats"][ch]["median"]) for ch in "RGB"}
        assert len({round(v, 4) for v in med.values()}) == 3, med

        for scale in (2.0, 3.0):
            frame = (frame1 * scale).astype(np.float32)
            _option_a(win, frame, f"f{scale}")

            model = win._histogram_model
            assert model is not None
            assert model is win.right_histogram_view.model
            assert model["channels"] == ["R", "G", "B"]

            for ch in ("R", "G", "B"):
                counts = model["counts"][ch]
                assert counts.shape == (512,)
                top_frac = _top_bin_fraction(model, ch)
                med = float(model["stats"][ch]["median"])
                mean = float(model["stats"][ch]["mean"])

                # Regression witness: before the drift fix the entire population
                # collapsed into the top bin (x = 1) with med/mean == 1.0.
                assert top_frac < 0.5, (
                    f"scale={scale} {ch}: histogram top bin collapsed "
                    f"(frac={top_frac:.4f})"
                )
                assert med < 0.9, (
                    f"scale={scale} {ch}: histogram median collapsed to {med:.4f}"
                )
                assert mean < 0.9, (
                    f"scale={scale} {ch}: histogram mean collapsed to {mean:.4f}"
                )
                assert 0.0 < med < 1.0 and 0.0 < mean < 1.0, (
                    f"scale={scale} {ch}: median={med:.4f} mean={mean:.4f}"
                )
    finally:
        win.shutdown()
