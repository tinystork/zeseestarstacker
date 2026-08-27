"""ZSSS-OTPUX-STABLE-B — preview view-state persistence across valid previews.

Focused offscreen tests proving that a *valid* backend preview replaces only
the displayed scientific content, while the user's accumulated view state
(continuous zoom factor / zoom-combo semantic state, pan offsets and rotation)
survives:

* valid successive backend preview N -> N+1 preserves zoom / pan / rotation,
  and the rendered geometry reflects it;
* a same-scientific-identity duplicate / resolution refresh also preserves all
  three (and stays inert for STABLE-A identity/`raw_revision`);
* a genuinely new scientific preview N+1 preserves view state while live auto
  still fires exactly once per new batch;
* explicit lifecycle boundaries reset view state: `_on_run_started` (new run),
  a successful new initial-folder preview, `_clear_preview`, and an
  invalid/unrenderable payload;
* ordinary view operations leave STABLE-A identity / counters unchanged.

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
    app = QApplication.instance()
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


def _histogram_idle(win: MainWindow) -> bool:
    """True once the Option-A float histogram model is applied and the bounded
    coordinator has no in-flight/pending work (a stable counter baseline)."""
    return (
        win._histogram_model is not None
        and win._histogram_model_revision == win._wb_only_revision
        and not win._histogram_coordinator.is_running
        and not win._histogram_coordinator.is_pending
    )


def _preview_uint8(width: int, height: int):
    """Build a ``width x height`` RGB uint8 array (width=W, height=H)."""
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Option-A helpers (for the STABLE-A identity / live-auto cross-check only)
# ---------------------------------------------------------------------------
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


def _mode_payload(mode: str, n: int, seed: int = 100) -> BackendPreviewPayload:
    raw = _raw(seed + n)
    if mode == "drizzle":
        header = {"PREV_SRC": "Drizzle Accumulator"}
        image_count, current_batch = n, n
    elif mode == "classic":
        header = {"PREV_SRC": "SUM/W Accumulators"}
        image_count, current_batch = n * 10, n
    else:  # reproject: legacy path carries no PREV_SRC
        header = {}
        image_count, current_batch = n * 10, n
    return BackendPreviewPayload(
        data=(_legacy(raw), raw),
        header=header,
        stack_name=f"{mode} {n}",
        image_count=image_count,
        total_images=100,
        current_batch=current_batch,
        total_batches=10,
    )


def _arm_run(win: MainWindow) -> None:
    win._running = True
    win._run_context_id += 1
    win._last_live_auto_batch_token = None
    win._live_auto_stretch_count = 0
    win._live_auto_wb_count = 0
    win._live_bp = None
    win._live_wp = None
    win._set_live_auto_stretch_enabled(True)
    win._set_live_auto_wb_enabled(True)


# ---------------------------------------------------------------------------
# (A) valid successive preview preserves zoom / pan / rotation + geometry
# ---------------------------------------------------------------------------
def test_valid_successive_preview_preserves_zoom_pan_rotation(qapp):
    win = MainWindow()
    try:
        # N: 20x10, user sets 200% zoom, 90° rotation, nonzero pan.
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="n"))
        win.zoom_combo.setCurrentText("200%")
        win.rotate_right_button.click()
        win._on_pan_delta(12.0, -5.0)
        assert win.preview_rotation == 90
        assert win._preview_zoom_factor == 2.0
        assert (win._view_offset_x, win._view_offset_y) == (12.0, -5.0)

        # N+1: 8x12 — a valid successive preview preserves all three.
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(8, 12), stack_name="n1"))
        assert win.preview_rotation == 90
        assert win._preview_zoom_factor == 2.0
        assert win.zoom_combo.currentText() == "200%"
        assert (win._view_offset_x, win._view_offset_y) == (12.0, -5.0)
        # Rendered geometry reflects the preserved rotation + zoom (8x12 → 24x16).
        assert win.resolution_label.text() == "8x12 → 24x16 · 200% · 90°"
    finally:
        win.shutdown()


def test_custom_wheel_zoom_survives_successive_preview(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="n"))
        # Custom (non-preset) continuous zoom: 100% -> ~115%, blank combo.
        win._on_wheel_zoom(1, 10.0, 5.0)
        assert win._preview_zoom_factor == pytest.approx(1.15)
        assert win.zoom_combo.currentText() == ""

        win._on_preview(BackendPreviewPayload(data=_preview_uint8(8, 12), stack_name="n1"))
        assert win._preview_zoom_factor == pytest.approx(1.15)
        assert win.zoom_combo.currentText() == ""
        assert "115%" in win.resolution_label.text()
    finally:
        win.shutdown()


def test_fit_mode_survives_successive_preview(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="n"))
        win.zoom_combo.setCurrentText("Fit")
        assert win.zoom_combo.currentText() == "Fit"

        win._on_preview(BackendPreviewPayload(data=_preview_uint8(8, 12), stack_name="n1"))
        assert win.zoom_combo.currentText() == "Fit"
        assert win.has_preview_image
        assert "Fit" in win.resolution_label.text()
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (B) same-identity duplicate / resolution refresh preserves view state + inert
# ---------------------------------------------------------------------------
def test_same_identity_resolution_refresh_preserves_view_state(qapp):
    win = MainWindow()
    try:
        # Option-A payloads preserve the frozen-anchor identity context across
        # a same-identity re-render (a legacy payload would reset analysis).
        win._on_preview(_mode_payload("classic", 1))
        win.zoom_combo.setCurrentText("200%")
        win.rotate_right_button.click()
        win._on_pan_delta(5.0, 5.0)
        rev = win.raw_revision
        assert rev == 1

        # Same scientific identity re-rendered (repaint / resolution refresh).
        win._on_preview(_mode_payload("classic", 1, seed=999))
        assert win.preview_rotation == 90
        assert win._preview_zoom_factor == 2.0
        assert (win._view_offset_x, win._view_offset_y) == (5.0, 5.0)
        # Identity unchanged -> raw_revision stays put (STABLE-A inert).
        assert win.raw_revision == rev
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (C) _on_run_started resets view state once before the next run
# ---------------------------------------------------------------------------
def test_run_started_resets_view_state_once(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="a"))
        win.zoom_combo.setCurrentText("200%")
        win.rotate_right_button.click()
        # Before run start: the transformed pixmap geometry + resolution label
        # reflect the retained source at 200% / 90° (20x10 -> 20x40).
        assert win.preview_rotation == 90
        assert win._preview_zoom_factor == 2.0
        assert win.zoom_combo.currentText() == "200%"
        pm = win.preview_image_label.pixmap()
        assert pm is not None
        assert (pm.width(), pm.height()) == (20, 40)
        assert win.resolution_label.text() == "20x10 → 20x40 · 200% · 90°"

        # Accumulate a pan offset so the reset also proves recentring.
        win._on_pan_delta(7.0, -3.0)
        assert (win._view_offset_x, win._view_offset_y) == (7.0, -3.0)

        rev_before = win.raw_revision
        ident_before = win.displayed_identity
        target_before = win.live_target_identity

        win._on_run_started()

        assert win.preview_rotation == 0
        assert win._preview_zoom_factor == 1.0
        assert win.zoom_combo.currentText() == "100%"
        assert (win._view_offset_x, win._view_offset_y) == (0.0, 0.0)
        # Immediate reconciliation: displayed pixmap + label agree with the
        # reset state (20x10 at 100% / 0°), not the stale 200% / 90°.
        pm = win.preview_image_label.pixmap()
        assert pm is not None
        assert (pm.width(), pm.height()) == (20, 10)
        assert win.resolution_label.text() == "20x10 → 20x10 · 100%"
        assert "90°" not in win.resolution_label.text()
        # The pure view reconciliation did not advance STABLE-A identity state.
        assert win.raw_revision == rev_before
        assert win.displayed_identity == ident_before
        assert win.live_target_identity == target_before
    finally:
        win.shutdown()


def test_run_started_reconciliation_preserves_option_a_identity_and_histogram(qapp):
    win = MainWindow()
    try:
        _arm_run(win)
        win._on_preview(_mode_payload("classic", 1))
        # Wait for the initial float histogram model to be applied and the
        # bounded coordinator to go idle, so the request/job counters form a
        # stable baseline before the reconciliation.
        assert _pump_until(lambda: _histogram_idle(win))
        assert win.raw_revision == 1
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1

        # Non-default view state on top of the Option-A preview (24x24 source).
        win.zoom_combo.setCurrentText("200%")
        win.rotate_right_button.click()
        win._on_pan_delta(7.0, -3.0)
        assert win.preview_rotation == 90

        rev_before = win.raw_revision
        requests_before = win._histogram_compute_count
        jobs_before = win._histogram_coordinator.jobs_started

        win._on_run_started()

        # View state + displayed pixmap/label reconciled to defaults (0°/100%).
        assert win.preview_rotation == 0
        assert win._preview_zoom_factor == 1.0
        assert win.zoom_combo.currentText() == "100%"
        assert (win._view_offset_x, win._view_offset_y) == (0.0, 0.0)
        pm = win.preview_image_label.pixmap()
        assert pm is not None
        assert (pm.width(), pm.height()) == (24, 24)
        assert win.resolution_label.text() == "24x24 → 24x24 · 100%"
        # STABLE-A identity / raw_revision are not advanced by the pure view
        # reconciliation; live auto is reset by run start and never re-fired.
        assert win.raw_revision == rev_before
        assert win.live_target_identity is None
        assert win.live_auto_wb_count == 0
        assert win.live_auto_stretch_count == 0
        # Zero new histogram compute decision/job for the unchanged source.
        assert win._histogram_compute_count == requests_before
        assert win._histogram_coordinator.jobs_started == jobs_before
    finally:
        win.shutdown()


def test_run_started_without_preview_is_safe(qapp):
    win = MainWindow()
    try:
        # No retained preview: _on_run_started must stay safe and leave the
        # view controls disabled (nothing to reconcile).
        win._on_run_started()
        assert win.is_running is True
        assert not win.has_preview_image
        assert not win.zoom_combo.isEnabled()
        assert not win.rotate_left_button.isEnabled()
        assert not win.rotate_right_button.isEnabled()
        assert win.resolution_label.text() == "—"
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (D) invalid payload and _clear_preview reset view state + disable/clear
# ---------------------------------------------------------------------------
def test_invalid_payload_and_clear_reset_view_state(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="a"))
        win.zoom_combo.setCurrentText("200%")
        win.rotate_right_button.click()
        win._on_pan_delta(7.0, -3.0)

        # Invalid payload resets view state and disables the controls.
        win._on_preview(BackendPreviewPayload(data="garbage", stack_name="bad"))
        assert win.preview_rotation == 0
        assert win._preview_zoom_factor == 1.0
        assert (win._view_offset_x, win._view_offset_y) == (0.0, 0.0)
        assert not win.has_preview_image
        assert not win.zoom_combo.isEnabled()
        assert not win.rotate_left_button.isEnabled()
        assert not win.rotate_right_button.isEnabled()

        # Re-establish a valid preview, then clear it explicitly.
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="b"))
        win.zoom_combo.setCurrentText("200%")
        win.rotate_right_button.click()
        win._on_pan_delta(7.0, -3.0)
        assert win.preview_rotation == 90

        win._clear_preview("cleared")
        assert win.preview_rotation == 0
        assert win._preview_zoom_factor == 1.0
        assert (win._view_offset_x, win._view_offset_y) == (0.0, 0.0)
        assert not win.has_preview_image
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (E) successful new initial-folder preview resets view state
# ---------------------------------------------------------------------------
def test_initial_folder_success_resets_view_state(qapp, tmp_path, monkeypatch):
    import seestar.gui_qt.initial_preview as ip

    (tmp_path / "frame.fits").write_bytes(b"dummy")
    monkeypatch.setattr(
        ip,
        "load_initial_preview",
        lambda folder, filename, bayer: (np.full((4, 4), 0.5, np.float32), None),
    )

    win = MainWindow()
    try:
        # Establish a non-default view state on a prior preview.
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="old"))
        win.zoom_combo.setCurrentText("200%")
        win.rotate_right_button.click()
        win._on_pan_delta(7.0, -3.0)
        assert win.preview_rotation == 90

        win.input_edit.setText(str(tmp_path))
        win._try_show_first_input_image()

        # Wait for the async initial-preview result to land on the GUI thread.
        assert _pump_until(lambda: win._last_preview_folder is not None)

        # Successful new initial folder resets view state to defaults.
        assert win.preview_rotation == 0
        assert win._preview_zoom_factor == 1.0
        assert win.zoom_combo.currentText() == "100%"
        assert (win._view_offset_x, win._view_offset_y) == (0.0, 0.0)
        assert win.has_preview_image
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (F) STABLE-A counters/identity unchanged by ordinary view operations
# ---------------------------------------------------------------------------
def test_view_operations_do_not_advance_stable_a_identity(qapp):
    win = MainWindow()
    try:
        _arm_run(win)
        win._on_preview(_mode_payload("classic", 1))
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1
        rev = win.raw_revision
        ident = win.displayed_identity
        target = win.live_target_identity

        # Ordinary view operations (zoom / rotate / pan) change nothing about
        # the STABLE-A identity or live-auto counters.
        win._on_zoom_changed(0)
        win._on_rotate_left()
        win._on_rotate_right()
        win._on_pan_delta(12.0, 34.0)
        qapp.processEvents()

        assert win.raw_revision == rev
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1
        assert win.displayed_identity == ident
        assert win.live_target_identity == target
    finally:
        win.shutdown()


def test_successive_preview_preserves_view_and_fires_live_auto_once(qapp):
    win = MainWindow()
    try:
        _arm_run(win)
        win._on_preview(_mode_payload("classic", 1))
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1

        # Set a non-default view state, then a genuinely new batch N+1.
        win.zoom_combo.setCurrentText("200%")
        win.rotate_right_button.click()
        win._on_pan_delta(4.0, -2.0)

        win._on_preview(_mode_payload("classic", 2))

        # View state preserved across the genuinely new scientific preview...
        assert win.preview_rotation == 90
        assert win._preview_zoom_factor == 2.0
        assert (win._view_offset_x, win._view_offset_y) == (4.0, -2.0)
        # ...and live auto fired exactly once for the new batch (no duplicate).
        assert win.live_auto_wb_count == 2
        assert win.live_auto_stretch_count == 2
        assert win.raw_revision == 2
    finally:
        win.shutdown()
