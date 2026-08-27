"""ZSSS-OTPUX-STABLE-A — Live Auto freshness for Classic / Reproject / Drizzle.

Focused offscreen tests proving that Live Auto targets the *displayed*
scientific preview identity (derived from engine metadata, never the GUI
``drizzle_group_spin`` cadence) for all three preview modes:

* preview N -> live auto targets N; preview N+1 -> live auto targets N+1;
* a duplicate callback for the same identity performs no duplicate AutoWB /
  AutoStretch;
* on the same immutable WB-only buffer, live Auto Stretch and the one-shot Auto
  Stretch button match within control quantization;
* ordinary zoom / rotation / pan / resolution refresh / histogram detach /
  histogram zoom / repaint perform no live-auto work;
* manual BP/WP vs WB intent isolation, and input-payload / scientific-array
  isolation are preserved.

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

MODES = ["classic", "reproject", "drizzle"]


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
    """A deterministic RGB raw-linear float32 image with a red cast."""
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


def _payload_for(mode: str, n: int, raw: np.ndarray) -> BackendPreviewPayload:
    """Build a payload whose scientific preview identity is ``n``.

    Classic/Reproject use ``current_batch`` as the identity; Drizzle uses
    ``image_count`` (the accepted-frame counter).  ``PREV_SRC`` selects the mode.
    """
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


def _mode_payload(mode: str, n: int, seed: int = 100) -> BackendPreviewPayload:
    return _payload_for(mode, n, _raw(seed + n))


def _feed_and_wait(win: MainWindow, payload: BackendPreviewPayload) -> None:
    win._on_preview(payload)
    assert _pump_until(
        lambda: win._histogram_model is not None
        and win._histogram_model_revision == win._wb_only_revision
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
# (1) preview N -> auto target N; N+1 -> auto target N+1 (all three modes)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_live_auto_targets_each_new_preview(qapp, mode):
    win = MainWindow()
    try:
        _arm_run(win)

        _feed_and_wait(win, _mode_payload(mode, 1))
        assert win.preview_mode == mode
        assert win.displayed_identity[2] == 1
        assert win.live_target_identity == win.displayed_identity
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1
        assert win.raw_revision == 1

        _feed_and_wait(win, _mode_payload(mode, 2))
        assert win.displayed_identity[2] == 2
        assert win.live_target_identity == win.displayed_identity
        assert win.live_auto_wb_count == 2
        assert win.live_auto_stretch_count == 2
        assert win.raw_revision == 2
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (2) duplicate N -> no duplicate work (all three modes)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_duplicate_identity_performs_no_duplicate_work(qapp, mode):
    win = MainWindow()
    try:
        _arm_run(win)

        _feed_and_wait(win, _mode_payload(mode, 1))
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1
        rev = win.raw_revision

        # Same scientific identity again (repaint / resolution refresh).
        _feed_and_wait(win, _mode_payload(mode, 1, seed=999))
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1
        assert win.raw_revision == rev
        assert win.displayed_identity[2] == 1
        assert win.live_target_identity == win.displayed_identity
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (2b) Reproject batch N -> SUM/W-labelled same-batch refresh is inert
# ---------------------------------------------------------------------------
def test_reproject_batch_n_survives_sumw_resolution_refresh(qapp):
    """Regression: the real carrier-route transition must not change identity.

    Architect witness: a Reproject scientific preview is initially emitted by
    ``queue_manager._update_preview_master`` -> ``_update_preview`` with no
    ``PREV_SRC``; a real resolution refresh routes through
    ``queue_manager.refresh_preview`` -> ``_update_preview_sum_w`` and re-emits
    the same ``current_batch`` N with ``PREV_SRC="SUM/W Accumulators"``.  The
    route label may truthfully change (``reproject`` -> ``classic``) but the
    scientific identity ``("batch", N)`` must not: no duplicate live-auto work
    and no ``raw_revision`` advance.
    """
    win = MainWindow()
    try:
        _arm_run(win)

        # Reproject batch N via the legacy PREV_SRC-less master path.
        _feed_and_wait(win, _mode_payload("reproject", 1))
        assert win.preview_mode == "reproject"
        assert win.displayed_identity == (win._run_context_id, "batch", 1)
        assert win.live_target_identity == win.displayed_identity
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1
        rev = win.raw_revision

        # Same scientific batch N re-rendered via the SUM/W refresh route.
        _feed_and_wait(win, _mode_payload("classic", 1, seed=777))
        # The route label truthfully reports the SUM/W renderer...
        assert win.preview_mode == "classic"
        # ...but the scientific identity is unchanged -> inert.
        assert win.displayed_identity == (win._run_context_id, "batch", 1)
        assert win.live_target_identity == win.displayed_identity
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1
        assert win.raw_revision == rev
    finally:
        win.shutdown()


def test_reproject_n1_triggers_once_after_sumw_refresh(qapp):
    """After an inert SUM/W refresh of batch N, Reproject batch N+1 fires once."""
    win = MainWindow()
    try:
        _arm_run(win)

        _feed_and_wait(win, _mode_payload("reproject", 1))
        _feed_and_wait(win, _mode_payload("classic", 1, seed=777))  # inert refresh
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1

        # Reproject batch N+1 (a genuinely new scientific batch) -> exactly one pass.
        _feed_and_wait(win, _mode_payload("reproject", 2))
        assert win.preview_mode == "reproject"
        assert win.displayed_identity == (win._run_context_id, "batch", 2)
        assert win.live_target_identity == win.displayed_identity
        assert win.live_auto_wb_count == 2
        assert win.live_auto_stretch_count == 2
        assert win.raw_revision == 2
    finally:
        win.shutdown()


def test_cross_run_same_counter_triggers_once_per_run(qapp):
    """Cross-run reuse of the same counters fires once in the new run.

    Uses the production lifecycle seam ``_on_run_started`` (which bumps the run
    context and resets per-run dedupe/instrumentation) rather than the manual
    ``_arm_run`` helper.
    """
    win = MainWindow()
    try:
        # Run 1.
        win._on_run_started()
        _feed_and_wait(win, _mode_payload("classic", 1))
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1
        assert win.raw_revision == 1
        first_identity = win.displayed_identity
        assert first_identity == (win._run_context_id, "batch", 1)

        # Run 2 reuses the same batch counter; the run context disambiguates.
        win._on_run_started()
        _feed_and_wait(win, _mode_payload("classic", 1))
        assert win.live_auto_wb_count == 1  # reset per run, then re-fired once
        assert win.live_auto_stretch_count == 1
        assert win.raw_revision == 2  # monotonic across runs
        assert win.displayed_identity != first_identity
        assert win.displayed_identity == (win._run_context_id, "batch", 1)
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (3) same immutable WB-only buffer: live vs one-shot Auto Stretch equivalence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_live_vs_manual_auto_stretch_match_same_wb_only_buffer(qapp, mode):
    win = MainWindow()
    try:
        _arm_run(win)
        _feed_and_wait(win, _mode_payload(mode, 1))

        # Live Auto already derived the WB-only buffer and applied Auto Stretch.
        live_bp, live_wp = win.live_bp_wp
        assert live_bp is not None and live_wp is not None
        assert live_bp < live_wp

        wb_rev_before = win.wb_revision

        # One-shot Auto Stretch on the *same* immutable WB-only buffer must
        # reproduce the same BP/WP within control quantization, without
        # re-deriving the WB-only buffer (no WB change).
        win.auto_stretch_button.click()
        qapp.processEvents()

        assert win.wb_revision == wb_rev_before
        assert win.stretch_bp_spin.value() == pytest.approx(live_bp, abs=0.002)
        assert win.stretch_wp_spin.value() == pytest.approx(live_wp, abs=0.002)
        assert win.stretch_combo.currentText() == "asinh"
        # The one-shot button stays independent: it does not re-enable or
        # disable live auto, and it does not overwrite the live witness values.
        assert win.live_bp_wp == (live_bp, live_wp)
        assert win._live_auto_stretch_enabled
        assert win._live_auto_wb_enabled
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (4) ordinary view / histogram operations perform no live-auto work
# ---------------------------------------------------------------------------
def test_view_and_histogram_operations_do_no_live_auto_work(qapp):
    win = MainWindow()
    try:
        _arm_run(win)
        _feed_and_wait(win, _mode_payload("classic", 1))
        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1
        target = win.live_target_identity

        # zoom / rotation / pan / resolution refresh / histogram detach /
        # histogram zoom / ordinary repaint.
        win._on_zoom_changed(0)
        win._on_rotate_left()
        win._on_rotate_right()
        win._on_pan_delta(12.0, 34.0)
        win._on_preview_res_cycle()
        win._open_detached_histogram()
        win._zoom_histogram()
        win._refresh_preview_view()
        qapp.processEvents()

        assert win.live_auto_wb_count == 1
        assert win.live_auto_stretch_count == 1
        assert win.live_target_identity == target
        assert win.live_target_identity == win.displayed_identity
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (5) manual intent isolation: BP/WP disables only live stretch; WB only live WB
# ---------------------------------------------------------------------------
def test_manual_edits_isolate_live_auto_intent(qapp):
    win = MainWindow()
    try:
        _arm_run(win)
        _feed_and_wait(win, _mode_payload("classic", 1))

        # Manual BP edit -> disables only live stretch.
        new_bp = min(win._white_point - 0.1, win._black_point + 0.05)
        win.stretch_bp_spin.setValue(new_bp)
        qapp.processEvents()
        assert not win._live_auto_stretch_enabled
        assert win._live_auto_wb_enabled

        win._set_live_auto_stretch_enabled(True)

        # Manual WB edit -> disables only live WB.
        new_r = min(5.0, win.wb_r_spin.value() + 0.1)
        win.wb_r_spin.setValue(new_r)
        qapp.processEvents()
        assert not win._live_auto_wb_enabled
        assert win._live_auto_stretch_enabled
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (6) live auto never mutates input payload arrays nor scientific state
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_live_auto_never_mutates_input_arrays(qapp, mode):
    win = MainWindow()
    raw = _raw(500)
    legacy = _legacy(raw)
    raw_before = raw.copy()
    legacy_before = legacy.copy()
    try:
        _arm_run(win)
        _feed_and_wait(win, _payload_for(mode, 1, raw))

        assert np.array_equal(raw, raw_before)
        assert np.array_equal(legacy, legacy_before)
        # Retained analysis buffers are owned copies, never views into the
        # payload arrays (mutating them can never touch the input).
        assert not np.shares_memory(win._raw_linear, raw)
        assert not np.shares_memory(win._pristine_float, raw)
        assert not np.shares_memory(win._wb_only_float, raw)
        win._pristine_float[0, 0, 0] = 99.0
        assert np.array_equal(raw, raw_before)
    finally:
        win.shutdown()
