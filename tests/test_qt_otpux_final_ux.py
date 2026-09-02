"""OTPUX final UX addendum — detached histogram + batch-boundary live auto.

All tests are offscreen and exercise only Qt-owned display state.  No backend
science object, accumulator or input ndarray may be mutated.
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
    app = QApplication.instance()
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


def _raw(seed: int = 1, size: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.uniform(100.0, 300.0, size=(size, size))
    r = 1.35 * g
    b = 0.72 * g
    return np.stack((r, g, b), axis=-1).astype(np.float32)


def _legacy(raw: np.ndarray) -> np.ndarray:
    arr = raw.astype(np.float64)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _payload(
    raw: np.ndarray,
    *,
    batch: int | None = None,
    image_count: int | None = None,
    total_images: int | None = None,
    drizzle: bool = False,
) -> BackendPreviewPayload:
    header = {"PREV_SRC": "Drizzle Accumulator" if drizzle else "SUM/W Accumulators"}
    return BackendPreviewPayload(
        data=(_legacy(raw), raw),
        header=header,
        stack_name="OTPUX final UX",
        image_count=image_count,
        total_images=total_images,
        current_batch=batch,
        total_batches=8,
    )


def _feed_and_wait(win: MainWindow, payload: BackendPreviewPayload) -> None:
    win._on_preview(payload)
    assert _pump_until(
        lambda: win._histogram_model is not None
        and win._histogram_model_revision == win._wb_only_revision
    )


def _arm_run(win: MainWindow) -> None:
    win._running = True
    win._last_live_auto_batch_token = None
    win._live_auto_stretch_count = 0
    win._live_auto_wb_count = 0
    win._set_live_auto_stretch_enabled(True)
    win._set_live_auto_wb_enabled(True)


def test_detached_view_reuses_model_worker_and_stats_without_compute(qapp):
    win = MainWindow()
    try:
        _feed_and_wait(win, _payload(_raw(1)))
        compute_before = win._histogram_compute_count
        jobs_before = win._histogram_coordinator.jobs_started

        win.hist_expand_button.click()
        qapp.processEvents()
        detached = win._detached_histogram_window
        assert detached is not None and detached.isVisible()

        assert detached.histogram_view.model is win.right_histogram_view.model
        assert detached.histogram_view.model is win._histogram_model
        assert detached.histogram_view.stats is win.right_histogram_view.stats
        assert detached.stats_label.text() == win.right_histogram_status.text()
        assert win._histogram_compute_count == compute_before
        assert win._histogram_coordinator.jobs_started == jobs_before
        assert not hasattr(detached, "_histogram_coordinator")
    finally:
        win.shutdown()


def test_auto_actions_and_live_toggles_share_both_surfaces(qapp):
    win = MainWindow()
    try:
        _feed_and_wait(win, _payload(_raw(3)))
        detached = win._open_detached_histogram()

        # Detached one-shot AutoWB updates the sole MainWindow WB state; the
        # resulting authoritative model object is mirrored back into both views.
        detached.auto_wb_button.click()
        assert _pump_until(
            lambda: detached.histogram_view.model is win._histogram_model
            and win._histogram_model_revision == win._wb_only_revision
        )
        assert win.right_histogram_view.model is detached.histogram_view.model
        assert win._wb == (
            win.wb_r_spin.value(),
            win.wb_g_spin.value(),
            win.wb_b_spin.value(),
        )

        # Inline one-shot Auto Stretch updates the common BP/WP state and both
        # sets of markers without toggling either live-auto intent.
        win.auto_stretch_button.click()
        qapp.processEvents()
        assert detached.histogram_view.black_point == win._black_point
        assert detached.histogram_view.white_point == win._white_point
        assert win.right_histogram_view.black_point == win._black_point
        assert win._live_auto_stretch_enabled
        assert win._live_auto_wb_enabled

        detached.live_auto_wb_check.setChecked(False)
        assert not win.live_auto_wb_check.isChecked()
        win.live_auto_wb_check.setChecked(True)
        assert detached.live_auto_wb_check.isChecked()
        detached.live_auto_stretch_check.setChecked(False)
        assert not win.live_auto_stretch_check.isChecked()
        win.live_auto_stretch_check.setChecked(True)
        assert detached.live_auto_stretch_check.isChecked()
    finally:
        win.shutdown()


def test_bp_wp_zoom_and_close_reopen_are_synchronized(qapp):
    win = MainWindow()
    try:
        _feed_and_wait(win, _payload(_raw(2)))
        detached = win._open_detached_histogram()

        detached.histogram_view.rangeChanged.emit(0.200, 0.800)
        qapp.processEvents()
        assert (win._black_point, win._white_point) == (0.2, 0.8)
        assert win.right_histogram_view.black_point == 0.2
        assert detached.histogram_view.black_point == 0.2
        assert not win._live_auto_stretch_enabled
        assert win._live_auto_wb_enabled

        win._set_live_auto_stretch_enabled(True)
        win.right_histogram_view.rangeChanged.emit(0.300, 0.700)
        qapp.processEvents()
        assert detached.histogram_view.black_point == 0.3
        assert detached.histogram_view.white_point == 0.7

        win._zoom_histogram()
        assert detached.histogram_view.view_range == win.right_histogram_view.view_range
        win._reset_histogram_view()
        # PHI-R3: "reset view" returns to the model's full *preserved analysis
        # range* (0, upper); upper = max(1.0, finite max) is the analysis upper
        # bound declared by the float model (== 1.0 when no HDR headroom
        # exists).  Both presentation surfaces share the same range.
        model = win._histogram_model
        assert model is not None
        upper = float(model["range"][1])
        assert upper >= 1.0
        expected_full = (0.0, upper)
        assert detached.histogram_view.view_range == pytest.approx(expected_full)
        assert win.right_histogram_view.view_range == pytest.approx(expected_full)
        detached.close()
        qapp.processEvents()
        assert not detached.isVisible()
        reopened = win._open_detached_histogram()
        assert reopened is detached
        assert reopened.histogram_view.model is model
        assert reopened.histogram_view.black_point == win._black_point
        assert reopened.histogram_view.white_point == win._white_point
    finally:
        win.shutdown()


def test_live_auto_runs_once_per_classic_batch_and_not_on_other_previews(qapp):
    win = MainWindow()
    try:
        _arm_run(win)
        _feed_and_wait(
            win,
            _payload(_raw(10), batch=1, image_count=5, total_images=20),
        )
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1

        # Multiple callbacks/repaints for the same completed batch are deduped.
        _feed_and_wait(
            win,
            _payload(_raw(11), batch=1, image_count=5, total_images=20),
        )
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1

        # A preview lacking a positive completed-batch identity is not eligible.
        _feed_and_wait(win, _payload(_raw(12), batch=None, image_count=6, total_images=20))
        _feed_and_wait(win, _payload(_raw(13), batch=0, image_count=7, total_images=20))
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1

        _feed_and_wait(
            win,
            _payload(_raw(14), batch=2, image_count=10, total_images=20),
        )
        assert win.live_auto_wb_count == 2
        assert win.live_auto_stretch_count == 2
    finally:
        win.shutdown()


def test_live_auto_drizzle_targets_each_accepted_frame(qapp):
    win = MainWindow()
    try:
        _arm_run(win)
        # The drizzle_group_spin widget cadence is NOT the scientific freshness
        # authority: even with a large group size, every accepted frame is its
        # own scientific preview (``image_count``) and gets its own live-auto
        # pass targeting that same identity.
        win.drizzle_group_spin.setValue(50)

        _feed_and_wait(
            win,
            _payload(_raw(21), batch=1, image_count=1, total_images=3, drizzle=True),
        )
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1
        assert win.preview_mode == "drizzle"
        assert win.displayed_identity[2] == 1
        assert win.live_target_identity == win.displayed_identity

        _feed_and_wait(
            win,
            _payload(_raw(22), batch=2, image_count=2, total_images=3, drizzle=True),
        )
        assert win.live_auto_wb_count == 2
        assert win.live_auto_stretch_count == 2
        assert win.displayed_identity[2] == 2

        # A duplicate callback for the same accepted frame is inert.
        _feed_and_wait(
            win,
            _payload(_raw(23), batch=2, image_count=2, total_images=3, drizzle=True),
        )
        assert win.live_auto_wb_count == 2
        assert win.live_auto_stretch_count == 2

        _feed_and_wait(
            win,
            _payload(_raw(24), batch=3, image_count=3, total_images=3, drizzle=True),
        )
        assert win.live_auto_wb_count == 3
        assert win.live_auto_stretch_count == 3
        assert win.displayed_identity[2] == 3
    finally:
        win.shutdown()


def test_manual_edits_disable_only_the_corresponding_live_auto(qapp):
    win = MainWindow()
    try:
        _feed_and_wait(win, _payload(_raw(30)))
        win._set_live_auto_stretch_enabled(True)
        win._set_live_auto_wb_enabled(True)

        new_bp = min(win._white_point - 0.1, win._black_point + 0.05)
        win.stretch_bp_spin.setValue(new_bp)
        qapp.processEvents()
        assert not win._live_auto_stretch_enabled
        assert win._live_auto_wb_enabled
        assert not win.live_auto_stretch_check.isChecked()

        win._set_live_auto_stretch_enabled(True)
        new_r = min(5.0, win.wb_r_spin.value() + 0.1)
        win.wb_r_spin.setValue(new_r)
        qapp.processEvents()
        assert not win._live_auto_wb_enabled
        assert win._live_auto_stretch_enabled
        assert not win.live_auto_wb_check.isChecked()
    finally:
        win.shutdown()


def test_detached_and_live_display_operations_never_mutate_payload_arrays(qapp):
    win = MainWindow()
    raw = _raw(40)
    legacy = _legacy(raw)
    raw_before = raw.copy()
    legacy_before = legacy.copy()
    try:
        _arm_run(win)
        _feed_and_wait(
            win,
            BackendPreviewPayload(
                data=(legacy, raw),
                header={"PREV_SRC": "SUM/W Accumulators"},
                current_batch=1,
                image_count=5,
                total_images=5,
            ),
        )
        detached = win._open_detached_histogram()
        detached.auto_stretch_button.click()
        detached.auto_wb_button.click()
        detached.zoom_button.click()
        detached.histogram_view.rangeChanged.emit(0.15, 0.85)
        detached.close()
        qapp.processEvents()

        assert np.array_equal(raw, raw_before)
        assert np.array_equal(legacy, legacy_before)
        assert not np.shares_memory(win._raw_linear, raw)
        assert not np.shares_memory(win._pristine_float, raw)
        assert not np.shares_memory(win._wb_only_float, raw)
    finally:
        win.shutdown()
