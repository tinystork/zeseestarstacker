"""ZSSS-OTPUX-HIST-H1 — float histogram model + BP/WP live-drag contract tests.

Focused offscreen Qt tests for the H1 histogram boundary:

* the Option-A preview feeds the authoritative ``compute_histogram_float``
  model straight into ``HistogramView`` (512 deterministic bins, RGB/L, no
  ``QImage`` histogram round-trip);
* RGB/L ``log_counts`` presentation + per-channel stats correspond exactly to
  ``compute_histogram_float``;
* a WB change recomputes the histogram (and bumps the compute counter); BP/WP
  / stretch / gamma / B/C/S / zoom / rotation / pan neither change the model
  nor increment the computation counter;
* live drag coalesces intermediate ``rangeChanged`` emissions and always emits
  the exact final state on release;
* BP/WP handle <-> main-window control synchronisation and no crossing;
* Auto Stretch moves markers/state without recomputing the histogram;
* AutoWB causes the required histogram refresh and stays idempotent.

No real stacking, no engine, no Tk.  ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import BackendPreviewPayload, MainWindow, create_application
from seestar.gui_qt.histogram_view import HistogramView, format_histogram_stats
from seestar.gui_qt.preview_analysis import compute_histogram_float


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


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


def _option_a(win: MainWindow, raw: np.ndarray) -> None:
    win._on_preview(
        BackendPreviewPayload(data=(_legacy_normalize(raw), raw), stack_name="h1")
    )
    assert win._pristine_float is not None
    assert _wait_histogram(win)


def _pump_until(predicate, timeout_ms: int = 5000) -> bool:
    """Pump the Qt event loop until ``predicate`` is true (or time out).

    H2 moved the authoritative float histogram off the GUI thread; this helper
    replaces the previous synchronous-histogram assumption by draining queued
    worker results until the GUI-thread model/status has caught up.
    """
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


# ---------------------------------------------------------------------------
# (1) Option-A float model reaches the widget (512 bins, no QImage round-trip)
# ---------------------------------------------------------------------------
def test_option_a_model_reaches_widget_512_bins_no_qimage(qapp):
    win = MainWindow()
    try:
        raw = _red_cast_raw(size=16)
        _option_a(win, raw)

        view = win.right_histogram_view
        assert view.has_data
        model = view.model
        assert model is not None
        assert model["bins"] == 512
        assert model["channels"] == ["R", "G", "B"]

        # The widget model is the exact float-domain model (not a re-binned
        # 256-bin QImage) and is the same object main-window cached.
        wb_only = win._ensure_wb_only_float()
        expected = compute_histogram_float(wb_only)
        for ch in expected["counts"]:
            assert np.array_equal(model["counts"][ch], expected["counts"][ch])
            assert np.allclose(model["log_counts"][ch], expected["log_counts"][ch])
        assert model is win._histogram_model
        assert win._histogram_compute_count == 1
    finally:
        win.shutdown()


def test_option_a_histogram_keeps_sub_8bit_tonal_detail(qapp):
    """A 4096-level ramp stays >256 occupied bins: only the float path can."""
    win = MainWindow()
    try:
        ramp = np.linspace(0.01, 0.99, 4096, dtype=np.float32).reshape(4096, 1)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(ramp), ramp), stack_name="mono-ramp"
            )
        )
        assert _wait_histogram(win)
        model = win.right_histogram_view.model
        assert model is not None
        assert model["bins"] == 512
        assert model["channels"] == ["L"]
        occupied = int((model["counts"]["L"] > 0).sum())
        # An 8-bit QImage round-trip can never fill more than 256 distinct
        # bins; the float model fills (nearly) all 512.
        assert occupied > 256
        assert np.array_equal(
            model["counts"]["L"],
            compute_histogram_float(win._ensure_wb_only_float())["counts"]["L"],
        )
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (2) RGB/L log-count presentation + stats correspond to compute_histogram_float
# ---------------------------------------------------------------------------
def test_rgb_log_counts_and_stats_correspond(qapp):
    win = MainWindow()
    try:
        _option_a(win, _red_cast_raw(size=16))
        view = win.right_histogram_view
        model = view.model
        expected = compute_histogram_float(win._ensure_wb_only_float())

        assert model["channels"] == ["R", "G", "B"]
        for ch in ("R", "G", "B"):
            counts = model["counts"][ch]
            assert counts.dtype == np.int64
            assert counts.shape == (512,)
            assert np.allclose(
                model["log_counts"][ch],
                np.log1p(counts.astype(np.float64)),
            )
            assert model["stats"][ch] == expected["stats"][ch]
        # The widget exposes the same stats dict.
        assert view.stats == model["stats"]
        # Deterministic R/G/B-labelled stats string.
        label = format_histogram_stats(model["stats"])
        assert label is not None
        assert label.startswith("R ")
        assert "G " in label and "B " in label
        assert "mean" in label and "std" in label and "med" in label
    finally:
        win.shutdown()


def test_mono_model_L_channel(qapp):
    win = MainWindow()
    try:
        mono = np.linspace(0.1, 0.9, 32 * 32, dtype=np.float32).reshape(32, 32)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(mono), mono), stack_name="mono"
            )
        )
        assert _wait_histogram(win)
        model = win.right_histogram_view.model
        assert model is not None
        assert model["channels"] == ["L"]
        assert model["counts"]["L"].shape == (512,)
        assert set(model["stats"].keys()) == {"L"}
        for key in ("min", "max", "median", "mean", "std"):
            assert key in model["stats"]["L"]
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (3) WB change recomputes; other display controls do not
# ---------------------------------------------------------------------------
def test_wb_changes_histogram_but_display_controls_do_not(qapp):
    win = MainWindow()
    try:
        _option_a(win, _red_cast_raw(size=16))
        view = win.right_histogram_view
        base_counts = {k: v.copy() for k, v in view.histogram.items()}
        assert win._histogram_compute_count == 1

        # WB change -> recompute + different counts.
        win.wb_r_spin.setValue(0.5)
        assert _wait_histogram(win)
        assert win._histogram_compute_count == 2
        assert any(
            not np.array_equal(base_counts[k], view.histogram[k]) for k in base_counts
        )

        model_before = view.model
        counts_before = {k: v.copy() for k, v in view.histogram.items()}
        c = win._histogram_compute_count

        # Non-WB display controls: no recompute, identical model/counts.
        win.stretch_bp_spin.setValue(0.3)
        win.stretch_wp_spin.setValue(0.8)
        win.stretch_gamma_spin.setValue(1.3)
        win.stretch_combo.setCurrentText("log")
        win.brightness_spin.setValue(1.2)
        win.contrast_spin.setValue(1.1)
        win.saturation_spin.setValue(0.7)
        win._on_rotate_right()
        win._on_rotate_left()
        win._on_pan_delta(5.0, 5.0)
        win._on_wheel_zoom(1, 10.0, 10.0)

        assert win._histogram_compute_count == c
        assert view.model is model_before
        for k in counts_before:
            assert np.array_equal(counts_before[k], view.histogram[k])
    finally:
        win.shutdown()


def test_new_source_recomputes_histogram(qapp):
    win = MainWindow()
    try:
        _option_a(win, _red_cast_raw(size=16, seed=1))
        assert win._histogram_compute_count == 1
        model1 = win.right_histogram_view.model

        # A new raw source (successive preview) re-derives WB-only + recomputes.
        win._on_preview(
            BackendPreviewPayload(
                data=(
                    _legacy_normalize(_red_cast_raw(size=16, seed=2)),
                    _red_cast_raw(size=16, seed=2),
                ),
                stack_name="h1-b",
            )
        )
        assert _wait_histogram(win)
        assert win._histogram_compute_count == 2
        assert win.right_histogram_view.model is not model1
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (4) live drag: coalesced intermediates + exact final release
# ---------------------------------------------------------------------------
def _model_view() -> HistogramView:
    view = HistogramView()
    view.resize(256, 80)
    arr = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    view.set_model(compute_histogram_float(arr))
    return view


def test_live_drag_coalesces_and_emits_exact_final(qapp):
    view = _model_view()
    try:
        emitted = []
        view.rangeChanged.connect(lambda bp, wp: emitted.append((bp, wp)))

        x_bp = view._level_to_x(view.black_point)
        assert view._start_drag_at(x_bp) == "min"

        # Multiple raw moves coalesce into a single pending emission.
        view._drag_at(view._level_to_x(0.3))
        view._drag_at(view._level_to_x(0.4))
        view._drag_at(view._level_to_x(0.45))
        assert len(emitted) == 0

        # Timer flush emits exactly one coalesced update (latest value).
        view._emit_live_drag()
        assert len(emitted) == 1
        assert emitted[-1][0] == pytest.approx(view.black_point, abs=1e-9)

        # Another burst -> another single coalesced update.
        view._drag_at(view._level_to_x(0.6))
        view._drag_at(view._level_to_x(0.62))
        view._emit_live_drag()
        assert len(emitted) == 2

        # Final release emits the exact in-flight value (never dropped/rounded).
        view._drag_at(view._level_to_x(0.77))
        final_bp = view.black_point
        final_wp = view.white_point
        view._end_drag()
        assert len(emitted) == 3
        assert emitted[-1][0] == final_bp
        assert emitted[-1][1] == final_wp
        assert final_bp < final_wp
    finally:
        view.deleteLater()


def test_drag_release_emits_exactly_once_without_move(qapp):
    view = _model_view()
    try:
        emitted = []
        view.rangeChanged.connect(lambda bp, wp: emitted.append((bp, wp)))
        x_bp = view._level_to_x(view.black_point)
        view._start_drag_at(x_bp)
        view._end_drag()
        # A press+release without a move emits exactly once (no timer fire).
        assert len(emitted) == 1
    finally:
        view.deleteLater()


# ---------------------------------------------------------------------------
# (5) handle/control synchronisation + no crossing
# ---------------------------------------------------------------------------
def test_handle_and_control_synchronisation(qapp):
    win = MainWindow()
    try:
        _option_a(win, _red_cast_raw(size=16))
        c = win._histogram_compute_count

        # Handle -> control: a histogram rangeChanged mirrors into sliders.
        win.right_histogram_view.rangeChanged.emit(0.3, 0.7)
        assert win.stretch_bp_spin.value() == pytest.approx(0.3, abs=1e-3)
        assert win.stretch_wp_spin.value() == pytest.approx(0.7, abs=1e-3)
        # A BP/WP drag never recomputes the histogram.
        assert win._histogram_compute_count == c

        # Control -> handle: slider changes re-sync the histogram markers.
        win.stretch_bp_spin.setValue(0.2)
        win.stretch_wp_spin.setValue(0.6)
        assert win.right_histogram_view.black_point == pytest.approx(0.2, abs=1e-3)
        assert win.right_histogram_view.white_point == pytest.approx(0.6, abs=1e-3)
        assert win._histogram_compute_count == c
    finally:
        win.shutdown()


def test_handle_no_crossing_and_min_separation(qapp):
    view = _model_view()
    try:
        view.set_range(0.2, 0.8)
        # Drag the black-point handle past the white point -> no crossing.
        view._start_drag_at(view._level_to_x(view.black_point))
        view._drag_at(view._level_to_x(0.95))
        view._end_drag()
        assert view.black_point < view.white_point
        assert view.white_point - view.black_point >= 1e-4 - 1e-9

        # set_range enforces separation on inverted input.
        view.set_range(0.9, 0.1)
        assert 0.0 <= view.black_point < view.white_point <= 1.0
        assert view.white_point - view.black_point >= 1e-4 - 1e-9
    finally:
        view.deleteLater()


# ---------------------------------------------------------------------------
# (6) Auto Stretch moves markers without recomputing the histogram
# ---------------------------------------------------------------------------
def test_auto_stretch_updates_markers_without_recompute(qapp):
    win = MainWindow()
    try:
        _option_a(win, _red_cast_raw(size=32))
        model_before = win.right_histogram_view.model
        c = win._histogram_compute_count

        win.auto_stretch_button.click()

        assert win.stretch_combo.currentText() == "asinh"
        assert win.right_histogram_view.black_point == pytest.approx(
            win.stretch_bp_spin.value(), abs=1e-3
        )
        assert win.right_histogram_view.white_point == pytest.approx(
            win.stretch_wp_spin.value(), abs=1e-3
        )
        # No histogram recompute: same model object, counter unchanged.
        assert win._histogram_compute_count == c
        assert win.right_histogram_view.model is model_before
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (7) AutoWB refreshes the histogram and stays idempotent
# ---------------------------------------------------------------------------
def test_autowb_refreshes_histogram_and_is_idempotent(qapp):
    win = MainWindow()
    try:
        _option_a(win, _red_cast_raw(size=32))
        base = {k: v.copy() for k, v in win.right_histogram_view.histogram.items()}
        c = win._histogram_compute_count

        win.auto_wb_button.click()

        # A real red-cast correction, and the histogram was recomputed.
        assert _wait_histogram(win)
        assert win._wb != (1.0, 1.0, 1.0)
        assert win._wb[0] < 1.0
        assert win._histogram_compute_count == c + 1
        assert any(
            not np.array_equal(base[k], win.right_histogram_view.histogram[k])
            for k in base
        )

        model1 = win.right_histogram_view.model
        c2 = win._histogram_compute_count

        # Idempotent: same gains -> no WB change -> no further recompute.
        win.auto_wb_button.click()
        assert win._histogram_compute_count == c2
        assert win.right_histogram_view.model is model1
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (8) H1 corrective — adversarial contract tests
# ---------------------------------------------------------------------------
def _two_gain_cast_raw(size: int = 64, seed: int = 5) -> np.ndarray:
    """A raw-linear RGB cast where R is too bright and B is too dim, so AutoWB
    must change *both* the R and B gains (a two-gain correction)."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(80.0, 200.0, size=(size, size))
    r = g * 2.0
    b = g * 0.5
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _legacy_rgb(size: int = 16, seed: int = 7) -> np.ndarray:
    """A single-array uint8 RGB payload for the legacy QImage histogram path."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 255.0, size=(size, size, 3)).astype(np.uint8)


# (8.1) BP/WP crossing preserves BP<WP and exact agreement on every surface
def test_bp_wp_crossing_normalizes_and_agrees_everywhere(qapp):
    win = MainWindow()
    try:
        _option_a(win, _red_cast_raw(size=16))
        view = win.right_histogram_view

        # Spinboxes: BP up then WP down across it (edited WP clamps up).
        win.stretch_bp_spin.setValue(0.9)
        win.stretch_wp_spin.setValue(0.1)
        assert win.stretch_bp_spin.value() < win.stretch_wp_spin.value()
        assert win.stretch_bp_spin.value() == win._black_point
        assert win.stretch_wp_spin.value() == win._white_point
        assert view.black_point == win._black_point
        assert view.white_point == win._white_point
        assert win.stretch_wp_spin.value() == pytest.approx(0.901, abs=1e-6)
        assert win.stretch_wp_spin.value() - win.stretch_bp_spin.value() >= 0.001 - 1e-9

        # Reverse: WP down first, then BP up across it (edited BP clamps down).
        win.stretch_bp_spin.setValue(0.2)
        win.stretch_wp_spin.setValue(0.8)
        win.stretch_wp_spin.setValue(0.4)
        win.stretch_bp_spin.setValue(0.5)
        assert win.stretch_bp_spin.value() < win.stretch_wp_spin.value()
        assert win.stretch_bp_spin.value() == pytest.approx(0.399, abs=1e-6)
        assert win.stretch_bp_spin.value() == win._black_point
        assert win.stretch_wp_spin.value() == win._white_point
        assert view.black_point == win._black_point
        assert view.white_point == win._white_point

        # Sliders: drive the same crossing through the slider path.
        win.stretch_bp_slider.setValue(200)   # 0.2
        win.stretch_wp_slider.setValue(800)   # 0.8
        win.stretch_wp_slider.setValue(100)   # 0.1 -> crosses BP=0.2
        assert win.stretch_bp_spin.value() < win.stretch_wp_spin.value()
        assert win.stretch_bp_spin.value() == win._black_point
        assert win.stretch_wp_spin.value() == win._white_point
        assert view.black_point == win._black_point
        assert view.white_point == win._white_point
        assert win.stretch_wp_spin.value() == pytest.approx(0.201, abs=1e-6)

        # And the BP slider crossing in the other direction.
        win.stretch_bp_slider.setValue(200)
        win.stretch_wp_slider.setValue(800)
        win.stretch_bp_slider.setValue(900)   # 0.9 -> crosses WP=0.8
        assert win.stretch_bp_spin.value() < win.stretch_wp_spin.value()
        assert win.stretch_bp_spin.value() == pytest.approx(0.799, abs=1e-6)
        assert win.stretch_bp_spin.value() == win._black_point
        assert win.stretch_wp_spin.value() == win._white_point
        assert view.black_point == win._black_point
        assert view.white_point == win._white_point
    finally:
        win.shutdown()


# (8.2) set_range fail-closes / deterministically normalizes every bad input
def test_set_range_hardens_out_of_range_and_non_finite(qapp):
    view = _model_view()
    try:
        # Negative.
        view.set_range(-0.5, 0.3)
        assert 0.0 <= view.black_point < view.white_point <= 1.0

        # Above 1.
        view.set_range(0.2, 1.5)
        assert 0.0 <= view.black_point < view.white_point <= 1.0

        # Equal.
        view.set_range(0.5, 0.5)
        assert 0.0 <= view.black_point < view.white_point <= 1.0

        # Reversed.
        view.set_range(0.9, 0.1)
        assert 0.0 <= view.black_point < view.white_point <= 1.0

        # Non-finite (NaN / +inf / -inf) -> deterministic defaults, no invalid state.
        for bp, wp in (
            (float("nan"), 0.5),
            (0.2, float("inf")),
            (float("-inf"), float("nan")),
        ):
            view.set_range(bp, wp)
            assert 0.0 <= view.black_point < view.white_point <= 1.0

        # Deterministic normalization for the reversed pair.
        view.set_range(0.9, 0.1)
        assert view.black_point == pytest.approx(0.9)
        assert view.white_point == pytest.approx(0.901, abs=1e-6)
        assert view.white_point - view.black_point >= 0.001 - 1e-9
    finally:
        view.deleteLater()


# (8.3) legacy path: one compute on source/WB change, zero on display controls
def test_legacy_histogram_cached_across_display_controls(qapp, monkeypatch):
    import seestar.gui_qt.main_window as mw

    hist_calls = []
    stats_calls = []
    real_hist = mw.compute_histogram
    real_stats = mw.compute_histogram_stats

    def counting_hist(img, bins=256):
        hist_calls.append(1)
        return real_hist(img, bins=bins)

    def counting_stats(img):
        stats_calls.append(1)
        return real_stats(img)

    monkeypatch.setattr(mw, "compute_histogram", counting_hist)
    monkeypatch.setattr(mw, "compute_histogram_stats", counting_stats)

    win = MainWindow()
    try:
        # New legacy source -> exactly one histogram + one stats compute.
        win._on_preview(BackendPreviewPayload(data=_legacy_rgb(), stack_name="legacy"))
        assert win._pristine_float is None  # confirm the legacy path
        assert len(hist_calls) == 1
        assert len(stats_calls) == 1

        # WB change -> exactly one more of each.
        win.wb_r_spin.setValue(0.5)
        assert len(hist_calls) == 2
        assert len(stats_calls) == 2

        # Display controls -> zero additional histogram/stats compute.
        win.stretch_bp_spin.setValue(0.3)
        win.stretch_wp_spin.setValue(0.8)
        win.stretch_gamma_spin.setValue(1.3)
        win.stretch_combo.setCurrentText("log")
        win.brightness_spin.setValue(1.2)
        win.contrast_spin.setValue(1.1)
        win.saturation_spin.setValue(0.7)
        win._on_rotate_right()
        win._on_rotate_left()
        win._on_pan_delta(5.0, 5.0)
        win._on_wheel_zoom(1, 10.0, 10.0)
        assert len(hist_calls) == 2
        assert len(stats_calls) == 2
    finally:
        win.shutdown()


# (8.4) two-gain AutoWB is atomic; repeat is a no-op
def test_two_gain_autowb_is_atomic(qapp):
    win = MainWindow()
    try:
        _option_a(win, _two_gain_cast_raw(size=64))
        base = win._histogram_compute_count

        win.auto_wb_button.click()

        # Exactly one histogram compute for the explicit AutoWB action.
        assert win._histogram_compute_count == base + 1
        # A real two-gain correction: both R and B moved off neutral.
        assert win._wb[0] != 1.0
        assert win._wb[2] != 1.0

        # Second identical AutoWB -> zero additional compute.
        c2 = win._histogram_compute_count
        win.auto_wb_button.click()
        assert win._histogram_compute_count == c2
    finally:
        win.shutdown()


# (8.5) WB Reset is atomic (one compute) and a no-op at neutral
def test_wb_reset_atomic(qapp):
    win = MainWindow()
    try:
        _option_a(win, _red_cast_raw(size=32))

        # Multi-channel non-neutral WB.
        win.wb_r_spin.setValue(0.5)
        win.wb_g_spin.setValue(1.2)
        win.wb_b_spin.setValue(2.0)
        base = win._histogram_compute_count

        win.wb_reset_button.click()

        assert win._wb == (1.0, 1.0, 1.0)
        assert win._histogram_compute_count == base + 1

        # Reset at neutral -> zero recompute.
        c2 = win._histogram_compute_count
        win.wb_reset_button.click()
        assert win._histogram_compute_count == c2
    finally:
        win.shutdown()


# (8.6) a real histogram drag ends with handles/controls/state in agreement
def test_real_drag_final_agreement_across_surfaces(qapp):
    win = MainWindow()
    try:
        _option_a(win, _red_cast_raw(size=16))
        view = win.right_histogram_view
        view.resize(256, 80)

        x_bp = view._level_to_x(view.black_point)
        assert view._start_drag_at(x_bp) == "min"
        view._drag_at(view._level_to_x(0.6))
        view._drag_at(view._level_to_x(0.62))
        view._end_drag()

        bp = win.stretch_bp_spin.value()
        wp = win.stretch_wp_spin.value()
        assert bp < wp
        assert bp == win._black_point
        assert wp == win._white_point
        assert view.black_point == bp
        assert view.white_point == wp
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (9) H1 corrective #2 — live-drag grid agreement, RGB composite, robust X range
# ---------------------------------------------------------------------------
def _synthetic_model(x_range=(0.0, 1.0), channels=("R", "G", "B")):
    """A minimal float-model dict with a controllable ``x_range`` metadata."""
    bins = 512
    counts, log_counts, stats = {}, {}, {}
    for c in channels:
        arr = np.zeros(bins, dtype=np.int64)
        arr[200] = 3  # one non-empty bin so ``has_data`` is True
        counts[c] = arr
        log_counts[c] = np.log1p(arr.astype(np.float64))
        stats[c] = {"min": 0.0, "max": 1.0, "median": 0.5, "mean": 0.5, "std": 0.1}
    return {
        "bins": bins,
        "range": (0.0, 1.0),
        "channels": list(channels),
        "counts": counts,
        "log_counts": log_counts,
        "stats": stats,
        "x_range": x_range,
        "full_range": (0.0, 1.0),
    }


def _render_view_to_image(view):
    """Offscreen-render a ``HistogramView`` into an ARGB32 ``QImage``."""
    img = QImage(view.size(), QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 255))
    painter = QPainter(img)
    view.render(painter, QPoint(0, 0))
    painter.end()
    return img


# (9.1) live-drag: in-flight handle snaps to the control grid and every
# surface (handle/spin/slider/state) agrees exactly before release.
def test_live_drag_inflight_grid_agreement(qapp):
    win = MainWindow()
    try:
        _option_a(win, _red_cast_raw(size=16))
        view = win.right_histogram_view
        view.resize(256, 80)
        emitted = []
        view.rangeChanged.connect(lambda bp, wp: emitted.append((bp, wp)))

        x_bp = view._level_to_x(view.black_point)
        assert view._start_drag_at(x_bp) == "min"

        # Drag the black-point handle to a non-grid level and flush the live
        # coalesced emission seam (still mid-drag, before release).
        view._drag_at(view._level_to_x(0.33337))
        assert emitted == []  # coalesced: no per-move emission yet
        view._emit_live_drag()

        # The in-flight handle is snapped to the shared 0.001 control grid.
        assert view.black_point == 0.333  # == 333 / 1000.0 exactly
        assert view.black_point == 333 / 1000.0
        # Handle == spin == slider-derived == authoritative MainWindow state.
        assert win._black_point == view.black_point
        assert win.stretch_bp_spin.value() == view.black_point
        assert win.stretch_bp_slider.value() == 333
        assert win.stretch_bp_slider.value() * 0.001 == view.black_point
        # The live emission carried the same snapped value, and BP < WP holds.
        assert emitted[-1][0] == view.black_point
        assert emitted[-1][1] == view.white_point
        assert view.black_point < view.white_point
    finally:
        win.shutdown()


# (9.2) release still emits the exact final snapped value (no crossing/jitter).
def test_drag_release_emits_exact_final_snapped_value(qapp):
    win = MainWindow()
    try:
        _option_a(win, _red_cast_raw(size=16))
        view = win.right_histogram_view
        view.resize(256, 80)
        emitted = []
        view.rangeChanged.connect(lambda bp, wp: emitted.append((bp, wp)))

        x_bp = view._level_to_x(view.black_point)
        view._start_drag_at(x_bp)
        view._drag_at(view._level_to_x(0.33337))
        view._emit_live_drag()
        live_count = len(emitted)

        # A second move then release.
        view._drag_at(view._level_to_x(0.7777))
        final_bp = view.black_point
        final_wp = view.white_point
        view._end_drag()

        # Exactly one release emission (the live timer was stopped, no dupes).
        assert len(emitted) == live_count + 1
        assert emitted[-1][0] == final_bp
        assert emitted[-1][1] == final_wp
        # 0.7777 snaps to 0.778, no crossing, and every surface agrees.
        assert final_bp == 0.778
        assert final_bp == win._black_point
        assert final_wp == win._white_point
        assert win.stretch_bp_spin.value() == final_bp
        assert win.stretch_wp_spin.value() == final_wp
        assert final_bp < final_wp
        assert final_wp - final_bp >= 0.001 - 1e-9
    finally:
        win.shutdown()


# (9.3) RGB bars composite (not last-channel overpaint) and composition is
# restored for the BP/WP markers / axis labels.
def test_rgb_bars_are_composited_not_overpainted(qapp):
    view = HistogramView()
    try:
        view.resize(256, 96)
        bins = 512
        counts = {c: np.zeros(bins, dtype=np.int64) for c in ("R", "G", "B")}
        # R-only, G-only, B-only and a triple-overlap bin.
        counts["R"][64] = counts["R"][320] = 5
        counts["G"][128] = counts["G"][320] = 5
        counts["B"][192] = counts["B"][320] = 5
        log_counts = {c: np.log1p(counts[c].astype(np.float64)) for c in ("R", "G", "B")}
        stats = {
            c: {"min": 0.0, "max": 1.0, "median": 0.5, "mean": 0.5, "std": 0.1}
            for c in ("R", "G", "B")
        }
        view.set_model(
            {
                "bins": bins,
                "range": (0.0, 1.0),
                "channels": ["R", "G", "B"],
                "counts": counts,
                "log_counts": log_counts,
                "stats": stats,
                "x_range": (0.0, 1.0),
                "full_range": (0.0, 1.0),
            }
        )

        img = _render_view_to_image(view)
        rect = view._plot_rect()
        y = int(rect.top() + (rect.bottom() - rect.top()) * 0.5)

        def px(bin_idx):
            center = (bin_idx + 0.5) / bins
            return img.pixelColor(int(round(view._level_to_x(center))), y)

        # Single-channel regions stay distinguishable (additive + background).
        r = px(64)
        g = px(128)
        b = px(192)
        assert r.red() > r.blue() and r.red() > r.green()          # red-dominant
        assert g.green() > g.red() and g.green() > g.blue()        # green-dominant
        assert b.blue() > b.red() and b.blue() > b.green()         # blue-dominant

        # Triple overlap is a bright composite (R+G+B -> white), NOT the last
        # pure B channel (which would be ~(70, 120, 225)).
        triple = px(320)
        assert triple.red() > 200 and triple.green() > 200 and triple.blue() > 200
        assert not (triple.red() == 70 and triple.green() == 120 and triple.blue() == 225)
    finally:
        view.deleteLater()


def test_paint_bars_restores_composition_for_markers(qapp):
    """``_paint_bars`` must restore the painter composition state it changed,
    so the BP/WP markers and axis labels are drawn with the caller's mode."""
    view = HistogramView()
    try:
        view.resize(256, 80)
        view.set_model(_synthetic_model())
        img = QImage(64, 64, QImage.Format.Format_ARGB32)
        painter = QPainter(img)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Screen)
        view._paint_bars(painter, view._plot_rect(), view._model["log_counts"])
        assert (
            painter.compositionMode()
            == QPainter.CompositionMode.CompositionMode_Screen
        )
        painter.end()
    finally:
        view.deleteLater()


# (9.4) zoom/auto-zoom use the robust x_range; invalid ranges fall back to
# full [0, 1]; reset view/zoom return to full [0, 1].
def test_zoom_uses_robust_x_range_for_model(qapp):
    view = HistogramView()
    try:
        view.resize(256, 80)
        view.set_model(_synthetic_model(x_range=(0.2, 0.8)))
        view.zoom_histogram()
        assert view.view_range[0] == pytest.approx(0.2)
        assert view.view_range[1] == pytest.approx(0.8)

        # Auto-zoom path uses the same robust range.
        view.reset_histogram_view()
        view.auto_zoom_enabled = True
        view.zoom_histogram()
        assert view.view_range[0] == pytest.approx(0.2)
        assert view.view_range[1] == pytest.approx(0.8)
    finally:
        view.deleteLater()


def test_zoom_falls_back_to_full_range_for_invalid_x_range(qapp):
    view = HistogramView()
    try:
        view.resize(256, 80)
        invalid = (
            None,
            (0.5, 0.5),          # degenerate (hi <= lo)
            (0.8, 0.2),          # inverted
            (float("nan"), 0.8),  # non-finite lo
            (0.2, float("inf")),  # non-finite hi
            (-0.1, 0.8),          # lo out of domain
            (0.2, 1.5),           # hi out of domain
            "nope",               # not a sequence
            (0.2,),               # wrong length
        )
        for bad in invalid:
            view.set_model(_synthetic_model(x_range=bad))
            view.zoom_histogram()
            assert view.view_range == (0.0, 1.0), repr(bad)
    finally:
        view.deleteLater()


def test_reset_returns_full_range_for_model(qapp):
    view = HistogramView()
    try:
        view.resize(256, 80)
        view.set_model(_synthetic_model(x_range=(0.2, 0.8)))
        view.zoom_histogram()
        assert view.view_range[0] == pytest.approx(0.2)

        view.reset_histogram_view()
        assert view.view_range == (0.0, 1.0)

        view.zoom_histogram()
        view.reset_zoom()
        assert view.view_range == (0.0, 1.0)
    finally:
        view.deleteLater()


def test_zoom_widens_degenerate_narrow_robust_range(qapp):
    view = HistogramView()
    try:
        view.resize(256, 80)
        view.set_model(_synthetic_model(x_range=(0.5, 0.51)))
        view.zoom_histogram()
        lo, hi = view.view_range
        # Widen the 0.01-wide range to the deterministic 0.02 minimum width.
        assert hi - lo == pytest.approx(0.02)
        assert 0.0 <= lo < hi <= 1.0
    finally:
        view.deleteLater()
