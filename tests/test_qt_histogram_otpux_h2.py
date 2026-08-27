"""ZSSS-OTPUX-HIST-H2 — bounded latest-wins off-thread histogram lifecycle tests.

Focused offscreen Qt tests for the H2 histogram boundary:

* the authoritative ``compute_histogram_float`` runs on a non-GUI worker thread,
  and the model/status/widget application runs back on the GUI thread;
* a slow generation 1 + a newer generation 2 can never display generation 1
  last (latest-wins via an explicit monotonic generation token);
* a rapid N=1..many burst keeps <=1 running + <=1 pending, the pending is the
  latest, actual worker jobs stay bounded (first + latest) and the replacement
  counter records coalescing;
* a stale running result is discarded after a new source / WB change / reset,
  and the latest accepted model exactly matches the latest source;
* reset / invalid payload prevents an old result from repopulating the cleared
  UI;
* BP/WP / stretch / gamma / BCS / zoom / pan / rotation / Auto Stretch schedule
  zero jobs;
* manual WB and AutoWB schedule exactly one job when changed, and an idempotent
  repeat AutoWB schedules zero;
* clean shutdown stops the worker and blocks any late widget update;
* scientific/display buffers are never mutated by the worker analysis.

No real stacking, no engine, no Tk.  ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import BackendPreviewPayload, MainWindow, create_application
from seestar.gui_qt.preview_analysis import compute_histogram_float


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


def _red_cast_raw(size: int = 16, seed: int = 1) -> np.ndarray:
    """A raw-linear RGB image with a red cast (R = 1.4 * G, B = G)."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(100.0, 200.0, size=(size, size))
    r = g * 1.4
    b = g.copy()
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _option_a_payload(raw: np.ndarray) -> tuple:
    return (_legacy_normalize(raw), raw)


def _feed(win: MainWindow, raw: np.ndarray, name: str = "h2") -> None:
    win._on_preview(BackendPreviewPayload(data=_option_a_payload(raw), stack_name=name))


# ---------------------------------------------------------------------------
# (1) compute runs on a non-GUI worker thread
# ---------------------------------------------------------------------------
def test_compute_runs_on_non_gui_worker_thread(qapp):
    gui_thread = threading.current_thread()
    gui_qthread = qapp.thread()
    records = []

    def compute(buf):
        records.append((threading.current_thread(), QThread.currentThread()))
        return compute_histogram_float(buf)

    win = MainWindow(histogram_compute_fn=compute)
    try:
        _feed(win, _red_cast_raw(size=16))
        assert _wait_histogram(win)
        assert len(records) == 1
        py_thread, qt_thread = records[0]
        assert py_thread is not gui_thread
        assert qt_thread is not gui_qthread
        assert win._histogram_coordinator.has_live_thread
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (2) model/status/widget application runs on the GUI thread
# ---------------------------------------------------------------------------
def test_model_application_runs_on_gui_thread(qapp):
    gui_thread = threading.current_thread()
    applied_threads = []
    win = MainWindow()
    try:
        win._histogram_coordinator.result_ready.connect(
            lambda g, r, t: applied_threads.append(threading.current_thread())
        )
        _feed(win, _red_cast_raw(size=16))
        assert _wait_histogram(win)
        assert applied_threads, "no latest result was applied"
        assert all(t is gui_thread for t in applied_threads)
        # The widget holds the applied model.
        assert win.right_histogram_view.model is win._histogram_model
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (3) slow generation 1 + newer generation 2 cannot display generation 1 last
# ---------------------------------------------------------------------------
def test_slow_gen1_newer_gen2_cannot_show_gen1_last(qapp):
    gate = threading.Event()
    calls = []

    def compute(buf):
        calls.append(buf)
        if len(calls) == 1:
            gate.wait()  # block the first computation
        return compute_histogram_float(buf)

    win = MainWindow(histogram_compute_fn=compute)
    try:
        _feed(win, _red_cast_raw(size=16, seed=1))  # gen 1 (blocked)
        assert _pump_until(lambda: len(calls) == 1)
        # A newer WB change while gen 1 is still running -> gen 2 pending.
        win.wb_r_spin.setValue(0.5)
        assert win._histogram_coordinator.pending_generation is not None

        gate.set()  # release gen 1; it must be discarded, gen 2 applied
        assert _wait_histogram(win)

        # The applied model is exactly the gen-2 (WB 0.5) result — gen 1 never
        # overwrote it.
        model = win.right_histogram_view.model
        expected = compute_histogram_float(win._ensure_wb_only_float())
        for ch in expected["counts"]:
            assert np.array_equal(model["counts"][ch], expected["counts"][ch])
        assert win._histogram_model_revision == win._wb_only_revision == 2
        assert win._histogram_coordinator.stale_discarded == 1
        assert win._histogram_coordinator.latest_applied == 1
    finally:
        gate.set()
        win.shutdown()


# ---------------------------------------------------------------------------
# (4) rapid N=1..many: <=1 running + <=1 pending, pending is latest, jobs bounded
# ---------------------------------------------------------------------------
def test_rapid_burst_bounded_latest_wins(qapp):
    gate = threading.Event()

    def compute(buf):
        gate.wait()  # hold every dispatched job until released
        return compute_histogram_float(buf)

    win = MainWindow(histogram_compute_fn=compute)
    try:
        _feed(win, _red_cast_raw(size=16))  # gen 1 (running, blocked)
        assert _pump_until(lambda: win._histogram_coordinator.jobs_started == 1)
        coord = win._histogram_coordinator

        # Rapid WB revisions N=2..11 while gen 1 is blocked.
        for i in range(10):
            win.wb_r_spin.setValue(0.3 + 0.05 * i)

        # Bounded: exactly one running, at most one pending, pending is latest.
        assert coord.running_generation == 1
        assert coord.pending_generation is not None
        assert coord.pending_generation == coord.generation
        assert coord.requests_scheduled == 11  # 1 source + 10 WB revisions
        assert coord.jobs_started == 1          # only the first was dispatched
        # 10 schedules while busy -> first pending + 9 replacements (coalescing).
        assert coord.pending_replaced == 9

        gate.set()  # gen 1 finishes (stale), latest pending runs and applies
        assert _wait_histogram(win)

        assert coord.jobs_started == 2          # first + latest only
        assert coord.stale_discarded == 1
        assert coord.latest_applied == 1
        assert coord.running_generation is None
        assert coord.pending_generation is None
    finally:
        gate.set()
        win.shutdown()


# ---------------------------------------------------------------------------
# (5) stale running result discarded after a new source; latest source wins
# ---------------------------------------------------------------------------
def test_new_source_discards_stale_running_result(qapp):
    gate = threading.Event()

    def compute(buf):
        gate.wait()
        return compute_histogram_float(buf)

    win = MainWindow(histogram_compute_fn=compute)
    try:
        _feed(win, _red_cast_raw(size=16, seed=1))  # gen 1 (blocked)
        assert _pump_until(lambda: win._histogram_coordinator.jobs_started == 1)
        # A brand-new source while gen 1 is blocked -> gen 2 pending.
        _feed(win, _red_cast_raw(size=16, seed=2), name="h2-b")
        assert win._histogram_coordinator.pending_generation is not None

        gate.set()
        assert _wait_histogram(win)

        # Latest accepted model exactly matches the latest (seed-2) source.
        model = win.right_histogram_view.model
        expected = compute_histogram_float(win._ensure_wb_only_float())
        for ch in expected["counts"]:
            assert np.array_equal(model["counts"][ch], expected["counts"][ch])
        assert win._histogram_coordinator.stale_discarded == 1
    finally:
        gate.set()
        win.shutdown()


# ---------------------------------------------------------------------------
# (6) reset / invalid payload prevents an old result repopulating cleared UI
# ---------------------------------------------------------------------------
def test_reset_invalid_payload_blocks_old_result(qapp):
    gate = threading.Event()

    def compute(buf):
        gate.wait()
        return compute_histogram_float(buf)

    win = MainWindow(histogram_compute_fn=compute)
    try:
        _feed(win, _red_cast_raw(size=16))  # gen 1 (blocked)
        assert _pump_until(lambda: win._histogram_coordinator.jobs_started == 1)

        # Invalid payload clears the analysis context and invalidates the worker.
        win._on_preview(BackendPreviewPayload(data=None, stack_name="none"))
        assert not win.right_histogram_view.has_data
        assert win._histogram_model is None

        gate.set()  # old gen 1 finishes after the reset
        # Let queued events drain; the stale result must not repopulate the UI.
        assert _pump_until(lambda: win._histogram_coordinator.stale_discarded >= 1)
        assert not win.right_histogram_view.has_data
        assert win._histogram_model is None
    finally:
        gate.set()
        win.shutdown()


# ---------------------------------------------------------------------------
# (7) BP/WP/stretch/gamma/BCS/zoom/pan/rotation/Auto Stretch schedule zero jobs
# ---------------------------------------------------------------------------
def test_display_controls_schedule_zero_jobs(qapp):
    win = MainWindow()
    try:
        _feed(win, _red_cast_raw(size=16))
        assert _wait_histogram(win)
        coord = win._histogram_coordinator
        req_before = coord.requests_scheduled
        jobs_before = coord.jobs_started

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
        win.auto_stretch_button.click()

        assert coord.requests_scheduled == req_before
        assert coord.jobs_started == jobs_before
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (8) manual WB and AutoWB schedule exactly one when changed; idempotent = zero
# ---------------------------------------------------------------------------
def test_wb_and_autowb_schedule_exactly_one_when_changed(qapp):
    win = MainWindow()
    try:
        _feed(win, _red_cast_raw(size=32))
        assert _wait_histogram(win)
        coord = win._histogram_coordinator

        c = coord.requests_scheduled
        win.wb_r_spin.setValue(0.4)  # manual WB change -> exactly one
        assert coord.requests_scheduled == c + 1
        assert _wait_histogram(win)

        c2 = coord.requests_scheduled
        win.auto_wb_button.click()  # AutoWB changes the gains -> exactly one
        assert coord.requests_scheduled == c2 + 1
        assert _wait_histogram(win)
        assert win._wb != (1.0, 1.0, 1.0)

        c3 = coord.requests_scheduled
        win.auto_wb_button.click()  # idempotent repeat -> zero
        assert coord.requests_scheduled == c3
        assert coord.jobs_started == coord.requests_scheduled
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# (9) clean shutdown stops the worker and no widget update happens afterward
# ---------------------------------------------------------------------------
def test_clean_shutdown_stops_worker_and_blocks_late_result(qapp):
    gate = threading.Event()

    def compute(buf):
        gate.wait()
        return compute_histogram_float(buf)

    win = MainWindow(histogram_compute_fn=compute)
    try:
        _feed(win, _red_cast_raw(size=16))  # gen 1 (blocked)
        assert _pump_until(lambda: win._histogram_coordinator.jobs_started == 1)
        assert win._histogram_coordinator.has_live_thread
        assert win.right_histogram_view.model is None  # never applied

        # Shutdown while the worker is blocked: may not complete within 200 ms.
        win.shutdown(wait_ms=200)
        gate.set()  # release the blocked worker

        # A second shutdown joins the now-finishing worker.
        assert win.shutdown(wait_ms=5000) is True
        assert not win._histogram_coordinator.has_live_thread

        # No widget update happened: the stale gen-1 result never repopulated UI.
        assert win.right_histogram_view.model is None
        assert not win.right_histogram_view.has_data
    finally:
        gate.set()
        win.shutdown()


# ---------------------------------------------------------------------------
# (10) timed-out shutdown is retryable after the worker finishes naturally
# ---------------------------------------------------------------------------
def test_shutdown_timeout_retry_safe_after_natural_finish(qapp):
    """The H2 teardown lifecycle is safe across a timed-out first shutdown.

    A short-timeout shutdown must retain a valid, retryable QThread wrapper even
    after the worker later finishes and the GUI event loop runs.  ``has_live_thread``
    must never raise (the QThread C++ object is not auto-deleted while
    ``self._thread`` references it), and a retried shutdown must fully stop the
    thread, clear running/pending state, and leave the UI untouched.
    """
    entered = threading.Event()
    gate = threading.Event()

    def compute(buf):
        entered.set()          # prove compute is executing (entered)
        gate.wait()            # block until released
        return compute_histogram_float(buf)

    win = MainWindow(histogram_compute_fn=compute)
    try:
        _feed(win, _red_cast_raw(size=16))  # gen 1 (running, blocked)
        coord = win._histogram_coordinator

        # (1) prove compute is executing before the first shutdown.
        assert entered.wait(timeout=5.0), "worker never entered compute"
        assert coord.is_running is True

        # (2) first short-timeout shutdown returns False (non-destructive).
        assert win.shutdown(wait_ms=10) is False
        assert coord.has_live_thread is True  # still blocked, retryable

        # (3) release the gate and pump Qt events before the retry.
        gate.set()
        assert _pump_until(lambda: not coord.has_live_thread, timeout_ms=5000)

        # (4) has_live_thread is safe (no RuntimeError) and now False.
        assert coord.has_live_thread is False

        # (5) retry returns True; thread/worker ownership clean; running/pending
        #     cleared; UI remains clear (no post-shutdown widget update).
        assert win.shutdown(wait_ms=5000) is True
        assert coord.has_live_thread is False
        assert coord.is_running is False
        assert coord.is_pending is False
        assert win.right_histogram_view.model is None
        assert not win.right_histogram_view.has_data

        # (6) a third shutdown is idempotent and returns True.
        assert win.shutdown(wait_ms=5000) is True
    finally:
        gate.set()
        win.shutdown()


def test_normal_shutdown_clears_running_and_pending(qapp):
    """A successful shutdown clears ``is_running`` and ``is_pending``.

    When the worker finishes during the join (not a timed-out wait), the result
    channel is already disconnected, so ``_on_result`` can no longer clear
    ``_running``.  ``shutdown()`` must clear it (and the retained pending
    request) explicitly so a stopped coordinator is not left reporting a live
    computation.
    """
    entered = threading.Event()
    gate = threading.Event()

    def compute(buf):
        entered.set()
        gate.wait()
        return compute_histogram_float(buf)

    win = MainWindow(histogram_compute_fn=compute)
    try:
        _feed(win, _red_cast_raw(size=16))  # gen 1 (running, blocked)
        assert entered.wait(timeout=5.0)
        coord = win._histogram_coordinator
        assert coord.is_running is True

        # A newer WB request while gen 1 is blocked -> a retained pending request.
        win.wb_r_spin.setValue(0.5)
        assert coord.is_pending is True

        gate.set()  # release so the worker finishes during the join
        assert win.shutdown(wait_ms=5000) is True
        assert coord.has_live_thread is False
        assert coord.is_running is False
        assert coord.is_pending is False
        assert win.right_histogram_view.model is None
        assert not win.right_histogram_view.has_data
    finally:
        gate.set()
        win.shutdown()


# ---------------------------------------------------------------------------
# (11) scientific/display buffers are never mutated by the worker analysis
# ---------------------------------------------------------------------------
def test_worker_analysis_does_not_mutate_analysis_buffer(qapp):
    snapshots = []

    def compute(buf):
        before = buf.copy()
        result = compute_histogram_float(buf)
        snapshots.append((before, buf.copy()))
        return result

    win = MainWindow(histogram_compute_fn=compute)
    try:
        _feed(win, _red_cast_raw(size=16))
        assert _wait_histogram(win)
        assert len(snapshots) == 1
        before, after = snapshots[0]
        # The worker read the analysis buffer without mutating it.
        assert np.array_equal(before, after)
        assert np.array_equal(after, win._wb_only_float)
    finally:
        win.shutdown()


def test_scientific_and_display_buffers_unchanged_across_recompute(qapp):
    raw = _red_cast_raw(size=16, seed=9)
    raw_snapshot = raw.copy()
    legacy = _legacy_normalize(raw)
    legacy_snapshot = legacy.copy()

    win = MainWindow()
    try:
        win._on_preview(
            BackendPreviewPayload(data=(legacy, raw), stack_name="sci")
        )
        assert _wait_histogram(win)
        pristine_snapshot = win._pristine_float.copy()

        win.wb_r_spin.setValue(0.7)  # recompute off-thread
        assert _wait_histogram(win)

        # The backend payload arrays and the pristine pre-WB buffer are untouched.
        assert np.array_equal(raw, raw_snapshot)
        assert np.array_equal(legacy, legacy_snapshot)
        assert np.array_equal(win._pristine_float, pristine_snapshot)
        # The WB-only buffer was *replaced* (re-derived) for the new WB, never
        # mutated in place: its recorded gains match the authoritative ``_wb``.
        assert win._wb_only_wb == win._wb
    finally:
        win.shutdown()
