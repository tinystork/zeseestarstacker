"""ZSSS-OTPUX-QT-DISPLAY-STATE-01 — Option-A display-state lifecycle tests.

Focused offscreen Qt tests for the frozen-anchor raw-linear preview integration
in :class:`seestar.gui_qt.MainWindow`:

* frozen p0.5/p99.5 anchors across successive Option-A previews (a fixed raw
  pixel maps identically when later-frame extrema change);
* production Qt derives the display from the *raw-linear* second element, never
  the deliberately misleading legacy-normalized first element;
* manual BP/WP / WB / gamma / BCS survive a new backend preview (no silent
  Auto Stretch / AutoWB recomputation on successive updates);
* explicit Auto Stretch matches ``compute_auto_stretch_float`` on the WB-only
  float buffer and leaves anchors / raw / pristine buffers untouched;
* explicit Auto WB matches ``compute_auto_wb_float`` from the pristine pre-WB
  buffer even when the current WB is non-neutral, is idempotent, and never
  moves the anchors;
* legacy single-array and mono payloads keep the existing QImage path;
* run start / new folder / clear / invalid payload reset anchors + buffers;
* input payload arrays are never mutated.

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
from seestar.gui_qt.preview_analysis import (
    apply_wb_float,
    compute_auto_stretch_float,
    compute_auto_wb_float,
)


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _pump_until(qapp: QApplication, predicate, timeout_ms: int = 5000) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return bool(predicate())


def _legacy_normalize(arr: np.ndarray) -> np.ndarray:
    """A deliberately misleading legacy-normalized copy (min/max -> [0, 1])."""
    arr64 = arr.astype(np.float64)
    mn = float(np.nanmin(arr64))
    mx = float(np.nanmax(arr64))
    return np.clip((arr64 - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)


def _red_cast_raw(size: int = 32, seed: int = 1) -> np.ndarray:
    """A raw-linear RGB image with a red cast (R = 1.4 * G, B = G)."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(100.0, 200.0, size=(size, size))
    r = g * 1.4
    b = g.copy()
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# (1) frozen anchors across successive previews
# ---------------------------------------------------------------------------
def test_frozen_anchors_across_successive_previews(qapp):
    win = MainWindow()
    try:
        rng = np.random.default_rng(11)
        frame1 = rng.uniform(1.0, 10.0, size=(32, 32, 3)).astype(np.float32)
        ref = (5, 7)
        frame1[ref] = 5.0  # fixed raw pixel (all channels)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(frame1), frame1), stack_name="f1"
            )
        )

        assert win._anchor_lo is not None
        assert win._anchor_hi is not None
        lo1, hi1 = win._anchor_lo, win._anchor_hi
        mapped_ref1 = win._pristine_float[ref].copy()
        gen1 = win._analysis_generation

        # Frame 2: the stack evolved (everything brighter) but the reference
        # pixel's raw-linear value is unchanged.
        frame2 = rng.uniform(5.0, 50.0, size=(32, 32, 3)).astype(np.float32)
        frame2[ref] = 5.0
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(frame2), frame2), stack_name="f2"
            )
        )

        # Anchors are frozen (immutable within one run/context) and the mapped
        # reference pixel is identical.
        assert (win._anchor_lo, win._anchor_hi) == (lo1, hi1)
        assert win._analysis_generation == gen1  # no reset on successive updates
        assert np.array_equal(win._pristine_float[ref], mapped_ref1)
    finally:
        win.shutdown()


def test_option_a_uses_raw_second_element_not_legacy_first(qapp):
    win = MainWindow()
    try:
        legacy = np.zeros((4, 4, 3), dtype=np.float32)
        legacy[:, :, 0] = 1.0  # deliberately misleading: pure red
        raw = np.zeros((4, 4, 3), dtype=np.float32)
        raw[:, :, 2] = 1.0  # truth: pure blue
        win._on_preview(BackendPreviewPayload(data=(legacy, raw), stack_name="oa"))

        color = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)
        assert color.blue() > 200
        assert color.red() < 20
        # The analysis buffers track the raw-linear source (blue), not legacy.
        assert win._pristine_float[0, 0, 2] > 0.9
        assert win._pristine_float[0, 0, 0] < 0.1
    finally:
        win.shutdown()


def test_option_a_never_mutates_input_payload_arrays(qapp):
    win = MainWindow()
    try:
        raw = _red_cast_raw(size=16)
        legacy = _legacy_normalize(raw)
        legacy_before = legacy.copy()
        raw_before = raw.copy()

        win._on_preview(BackendPreviewPayload(data=(legacy, raw), stack_name="mut"))

        assert np.array_equal(legacy, legacy_before)
        assert np.array_equal(raw, raw_before)
        # The retained buffers are owned copies (mutating them never touches the
        # payload).
        win._pristine_float[0, 0, 0] = 99.0
        assert np.array_equal(raw, raw_before)
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (2) manual adjustments survive a new backend preview (no silent auto)
# ---------------------------------------------------------------------------
def test_manual_adjustments_survive_new_backend_preview(qapp):
    win = MainWindow()
    try:
        rng = np.random.default_rng(21)
        raw1 = rng.uniform(1.0, 10.0, size=(16, 16, 3)).astype(np.float32)
        win._on_preview(
            BackendPreviewPayload(data=(_legacy_normalize(raw1), raw1), stack_name="a")
        )

        # Manual display adjustments.
        win.stretch_bp_spin.setValue(0.2)
        win.stretch_wp_spin.setValue(0.8)
        win.wb_r_spin.setValue(1.3)
        win.wb_g_spin.setValue(1.1)
        win.wb_b_spin.setValue(0.9)
        win.stretch_gamma_spin.setValue(1.4)
        win.brightness_spin.setValue(1.2)
        win.contrast_spin.setValue(1.1)
        win.saturation_spin.setValue(0.8)

        bp_before = win._black_point
        wp_before = win._white_point
        wb_before = win._wb
        gamma_before = win._gamma

        # A new backend preview must preserve every manual display parameter.
        raw2 = rng.uniform(5.0, 50.0, size=(16, 16, 3)).astype(np.float32)
        win._on_preview(
            BackendPreviewPayload(data=(_legacy_normalize(raw2), raw2), stack_name="b")
        )

        assert win._black_point == bp_before
        assert win._white_point == wp_before
        assert win._wb == wb_before
        assert win._gamma == gamma_before
        assert win._brightness == pytest.approx(1.2)
        assert win._contrast == pytest.approx(1.1)
        assert win._saturation == pytest.approx(0.8)
    finally:
        win.shutdown()


def test_no_automatic_auto_stretch_or_autowb_on_successive_updates(qapp):
    win = MainWindow()
    try:
        rng = np.random.default_rng(31)
        raw1 = rng.uniform(1.0, 10.0, size=(16, 16, 3)).astype(np.float32)
        win._on_preview(
            BackendPreviewPayload(data=(_legacy_normalize(raw1), raw1), stack_name="a")
        )

        # Establish a clearly non-default, non-auto state.
        win.stretch_bp_spin.setValue(0.25)
        win.wb_r_spin.setValue(1.5)
        bp_before = win._black_point
        wp_before = win._white_point
        wb_before = win._wb
        stretch_before = win.stretch_combo.currentText()

        raw2 = rng.uniform(5.0, 50.0, size=(16, 16, 3)).astype(np.float32)
        win._on_preview(
            BackendPreviewPayload(data=(_legacy_normalize(raw2), raw2), stack_name="b")
        )

        # No silent Auto Stretch / AutoWB recalc: every value is bit-identical.
        assert win._black_point == bp_before
        assert win._white_point == wp_before
        assert win._wb == wb_before
        assert win.stretch_combo.currentText() == stretch_before
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (3) explicit Auto Stretch / Auto WB use the float buffers
# ---------------------------------------------------------------------------
def test_explicit_auto_stretch_matches_float_core_and_leaves_buffers(qapp):
    win = MainWindow()
    try:
        raw = _red_cast_raw(size=32)
        win._on_preview(
            BackendPreviewPayload(data=(_legacy_normalize(raw), raw), stack_name="as")
        )

        anchors_before = (win._anchor_lo, win._anchor_hi)
        raw_before = win._raw_linear.copy()
        pristine_before = win._pristine_float.copy()

        # Non-neutral WB so WB-only != pristine (exercises the WB-only path).
        win.wb_r_spin.setValue(1.2)
        wb_only = win._ensure_wb_only_float()
        exp_bp, exp_wp = compute_auto_stretch_float(wb_only)

        win.stretch_combo.setCurrentText("linear")
        win.auto_stretch_button.click()

        assert win.stretch_combo.currentText() == "asinh"
        assert win.stretch_bp_spin.value() == pytest.approx(round(exp_bp, 4), abs=0.002)
        assert win.stretch_wp_spin.value() == pytest.approx(round(exp_wp, 4), abs=0.002)
        assert win.stretch_bp_spin.value() < win.stretch_wp_spin.value()

        # Frozen anchors / raw / pristine buffers are never mutated.
        assert (win._anchor_lo, win._anchor_hi) == anchors_before
        assert np.array_equal(win._raw_linear, raw_before)
        assert np.array_equal(win._pristine_float, pristine_before)
    finally:
        win.shutdown()


def test_explicit_auto_wb_matches_float_core_from_pre_wb_and_is_idempotent(qapp):
    win = MainWindow()
    try:
        raw = _red_cast_raw(size=32)
        win._on_preview(
            BackendPreviewPayload(data=(_legacy_normalize(raw), raw), stack_name="awb")
        )

        anchors_before = (win._anchor_lo, win._anchor_hi)
        pristine_before = win._pristine_float.copy()

        # A non-neutral current WB must NOT influence the AutoWB estimate, which
        # comes from the pristine pre-WB buffer.
        win.wb_r_spin.setValue(1.4)
        win.wb_g_spin.setValue(1.0)
        win.wb_b_spin.setValue(0.8)

        exp = compute_auto_wb_float(win._pristine_float)
        assert exp != (1.0, 1.0, 1.0)  # a real red-cast correction
        assert exp[0] < 1.0  # red too strong -> reduce red

        win.auto_wb_button.click()

        assert win._wb[0] == pytest.approx(exp[0], abs=0.01)
        assert win._wb[1] == pytest.approx(exp[1], abs=1e-6)
        assert win._wb[2] == pytest.approx(exp[2], abs=0.01)
        assert (win._anchor_lo, win._anchor_hi) == anchors_before
        assert np.array_equal(win._pristine_float, pristine_before)

        # The already-WB buffer would give a (different) estimate — proving the
        # estimate came from pre-WB, not already-WB.
        already_wb = apply_wb_float(win._pristine_float, (1.4, 1.0, 0.8))
        exp_wrong = compute_auto_wb_float(already_wb)
        assert exp_wrong != pytest.approx(exp)

        # Repeat explicit action is deterministic / idempotent.
        win.auto_wb_button.click()
        assert win._wb[0] == pytest.approx(exp[0], abs=0.01)
        assert win._wb[1] == pytest.approx(exp[1], abs=1e-6)
        assert win._wb[2] == pytest.approx(exp[2], abs=0.01)
        assert (win._anchor_lo, win._anchor_hi) == anchors_before
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (4) legacy single-array and mono payloads keep the QImage path
# ---------------------------------------------------------------------------
def test_legacy_single_array_and_mono_are_safe(qapp):
    win = MainWindow()
    try:
        arr = np.zeros((4, 4, 3), dtype=np.float32)
        arr[:, :, 1] = 1.0
        win._on_preview(BackendPreviewPayload(data=arr, stack_name="legacy"))
        assert win.has_preview_image
        assert win._pristine_float is None  # legacy path: no float analysis
        assert win._anchor_lo is None

        mono = np.full((4, 4), 0.5, dtype=np.float32)
        win._on_preview(BackendPreviewPayload(data=mono, stack_name="mono"))
        assert win.has_preview_image
        assert win._pristine_float is None
        assert win._anchor_lo is None

        # A tuple whose second element is not an image array is legacy too.
        display = np.zeros((4, 4, 3), dtype=np.float32)
        display[:, :, 1] = 1.0
        hist = np.array([0.1, 0.2, 0.3])
        win._on_preview(BackendPreviewPayload(data=(display, hist), stack_name="tuple"))
        assert win.has_preview_image
        assert win._pristine_float is None
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (5) context reset: run start / new folder / clear / invalid payload
# ---------------------------------------------------------------------------
def _establish_option_a(win: MainWindow) -> None:
    rng = np.random.default_rng(41)
    raw = rng.uniform(1.0, 10.0, size=(16, 16, 3)).astype(np.float32)
    win._on_preview(
        BackendPreviewPayload(data=(_legacy_normalize(raw), raw), stack_name="ok")
    )
    assert win._anchor_lo is not None
    assert win._pristine_float is not None


def test_run_start_resets_anchors_and_buffers(qapp):
    win = MainWindow()
    try:
        _establish_option_a(win)
        gen_before = win._analysis_generation

        win._on_run_started()

        assert win._anchor_lo is None
        assert win._anchor_hi is None
        assert win._raw_linear is None
        assert win._pristine_float is None
        assert win._wb_only_float is None
        assert win._analysis_generation == gen_before + 1
    finally:
        win.shutdown()


def test_new_folder_resets_anchors_at_load_start(qapp, tmp_path, monkeypatch):
    import seestar.gui_qt.initial_preview as ip

    (tmp_path / "frame.fits").write_bytes(b"dummy")
    monkeypatch.setattr(
        ip,
        "load_initial_preview",
        lambda folder, filename, bayer: (np.full((4, 4), 0.5, np.float32), None),
    )

    win = MainWindow()
    try:
        _establish_option_a(win)
        gen_before = win._analysis_generation

        win.input_edit.setText(str(tmp_path))
        win._try_show_first_input_image()

        # Reset happens synchronously at load start (before the async result).
        assert win._anchor_lo is None
        assert win._raw_linear is None
        assert win._pristine_float is None
        assert win._analysis_generation == gen_before + 1
    finally:
        win.shutdown()


def test_clear_preview_resets_anchors_and_buffers(qapp):
    win = MainWindow()
    try:
        _establish_option_a(win)
        gen_before = win._analysis_generation

        win._clear_preview("cleared")

        assert win._anchor_lo is None
        assert win._anchor_hi is None
        assert win._raw_linear is None
        assert win._pristine_float is None
        assert win._wb_only_float is None
        assert win._analysis_generation == gen_before + 1
        assert not win.has_preview_image
    finally:
        win.shutdown()


def test_invalid_payload_clears_stale_preview_analysis(qapp):
    win = MainWindow()
    try:
        _establish_option_a(win)
        gen_before = win._analysis_generation

        win._on_preview(BackendPreviewPayload(data="garbage", stack_name="bad"))

        assert win._anchor_lo is None
        assert win._anchor_hi is None
        assert win._raw_linear is None
        assert win._pristine_float is None
        assert win._wb_only_float is None
        assert win._analysis_generation == gen_before + 1
        assert not win.has_preview_image
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (6) the histogram is fed from the cached WB-only float buffer (Option-A)
# ---------------------------------------------------------------------------
def test_option_a_histogram_uses_cached_wb_only_float(qapp):
    win = MainWindow()
    try:
        raw = _red_cast_raw(size=16)
        win._on_preview(
            BackendPreviewPayload(data=(_legacy_normalize(raw), raw), stack_name="hist")
        )

        # H2: the authoritative float histogram is computed off the GUI thread;
        # wait for the queued worker result to be applied before asserting.
        assert _pump_until(
            qapp, lambda: win.right_histogram_view.has_data
        )
        assert win.right_histogram_view.has_data
        assert win._wb_only_float is not None

        # A WB change re-derives the WB-only float and the histogram reacts.
        base = {k: v.copy() for k, v in win.right_histogram_view.histogram.items()}
        win.wb_r_spin.setValue(0.5)
        assert _pump_until(
            qapp,
            lambda: win._histogram_model_revision == win._wb_only_revision,
        )
        after = win.right_histogram_view.histogram
        assert any(
            not np.array_equal(base[k], after[k]) for k in base
        ), "Option-A histogram did not react to white balance"
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (7) classifier dtype semantics: nonnumeric 2D second elements stay legacy
# ---------------------------------------------------------------------------
def test_option_a_classifier_accepts_only_numeric_float_int_uint():
    from seestar.gui_qt.main_window import _is_option_a_preview_payload

    legacy = np.zeros((2, 2), dtype=np.float32)

    # Numeric second elements that the float core ingests -> Option-A.
    for dtype in (np.float32, np.float64, np.int16, np.int32, np.uint8, np.uint16):
        second = np.ones((2, 2), dtype=dtype)
        assert _is_option_a_preview_payload((legacy, second)), dtype

    # 3D numeric stays accepted.
    assert _is_option_a_preview_payload(
        (legacy, np.ones((2, 2, 3), dtype=np.float32))
    )

    # Nonnumeric 2D second elements must NOT be classified Option-A: bool,
    # string/unicode, bytes, object and structured/void dtypes.
    nonnumeric = [
        np.ones((2, 2), dtype=bool),
        np.full((2, 2), "text", dtype="U4"),
        np.full((2, 2), b"x", dtype="S3"),
        np.zeros((2, 2), dtype=object),
        np.zeros((2, 2), dtype=[("a", "f4"), ("b", "f4")]),
    ]
    for second in nonnumeric:
        assert not _is_option_a_preview_payload((legacy, second)), second.dtype

    # A duck-typed object with ndim/shape but no dtype is also not Option-A.
    class ShapeOnly:
        ndim = 2
        shape = (2, 2)

    assert not _is_option_a_preview_payload((legacy, ShapeOnly()))


def test_nonnumeric_2d_second_element_follows_legacy_path(qapp):
    win = MainWindow()
    try:
        legacy = np.zeros((4, 4, 3), dtype=np.float32)
        legacy[:, :, 1] = 1.0  # a valid legacy green image (first element)

        nonnumeric_seconds = [
            np.ones((4, 4), dtype=bool),
            np.full((4, 4), "x", dtype="U1"),
            np.zeros((4, 4), dtype=object),
            np.zeros((4, 4), dtype=[("a", "f4")]),
        ]
        for second in nonnumeric_seconds:
            win._on_preview(
                BackendPreviewPayload(data=(legacy, second), stack_name="legacy-fb")
            )
            # Legacy render path only: an image is produced, but no raw /
            # pristine / anchor float analysis state is ever created, and the
            # payload does not crash the window.
            assert win.has_preview_image
            assert win._raw_linear is None
            assert win._pristine_float is None
            assert win._anchor_lo is None
            assert win._anchor_hi is None
    finally:
        win.shutdown()
