"""M22 tests: live Res-factor control to the engine preview downsample.

Backend E2E part 3 (checklist 16.15 backend half): the Qt Res button
(``preview_res_button``) must drive the live engine
``set_preview_downsample_factor`` + ``refresh_preview`` while a run is active,
through a thread-safe control channel (GUI thread -> ``RunController`` ->
``RunWorker`` -> backend -> stacker instance).  Idle clicks stay display-only.

Tests (engine-free, a faithful fake stacker):

1. mapping unit — the Qt cycle value (1/2/3/4) maps to the engine factor
   verbatim (identity: ``Res 1/1`` -> 1, ``Res 1/2`` -> 2, ``Res 1/4`` -> 4);
   missing stacker methods are ignored,
2. idle — the button cycles the label and never invokes the control channel,
3. active run — a Res click applies the factor to the live stacker instance
   (``preview_downsample_factor`` changes),
4. no-op — no backend / no stacker reachable -> no crash, no error,
5. thread-safety smoke — rapid Res clicks from the GUI thread while the worker
   thread runs the fake stacker -> factors applied in order, no race crash,
6. display-only regressions — the Res factor never leaks into backend kwargs and
   never mutates ``_preview_source``.

No real stacking, no engine, no Tk, no FITS/PNG writes.
``QT_QPA_PLATFORM=offscreen`` is set defensively before any ``QApplication``.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import BackendPreviewPayload, MainWindow, create_application
from seestar.gui_qt.backend_runner import (
    SeestarQueuedStackerBackend,
    SimulatedRunBackend,
)
from seestar.gui_qt.main_window import DEFAULT_PREVIEW_RES_FACTOR, PREVIEW_RES_FACTORS
from seestar.gui_qt.run_controller import RunController


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


@pytest.fixture()
def window(qapp):
    win = MainWindow()
    yield win
    win.shutdown()


def _pump_until(qapp: QApplication, predicate, timeout_ms: int = 5000) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return bool(predicate())


def _rgb(width: int, height: int, r: int, g: int, b: int) -> np.ndarray:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = r
    arr[:, :, 1] = g
    arr[:, :, 2] = b
    return arr


class _FakeLiveStacker:
    """Faithful fake exposing the live preview-downsample control surface.

    ``is_running`` stays ``True`` until ``stop()`` so the backend's polling loop
    keeps the run alive while the test clicks Res from the GUI thread.
    """

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = dict(kwargs)
        self.align_on_disk = None
        self.progress_cb = None
        self.preview_cb = None
        self.start_kwargs = None
        self._running = False
        self.stop_called = False
        self.preview_downsample_factor = 2  # engine default
        self.downsample_calls = []
        self.refresh_calls = 0
        self.gui_event_queue = None  # no deferred engine events

    def set_progress_callback(self, cb) -> None:
        self.progress_cb = cb

    def set_preview_callback(self, cb) -> None:
        self.preview_cb = cb

    def start_processing(self, **kwargs):
        self.start_kwargs = dict(kwargs)
        self._running = True
        return True

    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self.stop_called = True
        self._running = False

    def set_preview_downsample_factor(self, factor) -> None:
        self.downsample_calls.append(factor)
        self.preview_downsample_factor = factor

    def refresh_preview(self) -> None:
        self.refresh_calls += 1


# --------------------------------------------------------------------------
# (1) mapping unit: Qt cycle value -> engine factor (identity)
# --------------------------------------------------------------------------
def test_mapping_unit_qt_cycle_value_to_engine_factor_is_identity():
    """``Res 1/1`` -> 1, ``Res 1/2`` -> 2, ``Res 1/3`` -> 3, ``Res 1/4`` -> 4.

    The engine factor IS the 1/N denominator (its ``set_preview_downsample_factor``
    clamps to 1..4 and reports ``Preview resolution set to 1/{f}``), so no
    display-ratio conversion is needed: the Qt cycled value maps verbatim.
    """
    assert PREVIEW_RES_FACTORS == (1, 2, 3, 4)
    assert DEFAULT_PREVIEW_RES_FACTOR == 1

    class Recorder:
        def __init__(self) -> None:
            self.factors = []
            self.refreshes = 0

        def set_preview_downsample_factor(self, f) -> None:
            self.factors.append(f)

        def refresh_preview(self) -> None:
            self.refreshes += 1

    r = Recorder()
    for qt_value in PREVIEW_RES_FACTORS:
        SeestarQueuedStackerBackend._apply_preview_downsample_control(r, qt_value)
    assert r.factors == [1, 2, 3, 4]
    assert r.refreshes == 4


def test_mapping_unit_ignores_missing_stackers_methods():
    """A stacker missing either control method must not raise."""
    class NoMethods:
        pass

    SeestarQueuedStackerBackend._apply_preview_downsample_control(NoMethods(), 2)


# --------------------------------------------------------------------------
# (2) idle: button cycles the label, no engine call
# --------------------------------------------------------------------------
def test_idle_res_click_cycles_label_without_engine_call(qapp):
    constructed = []

    def factory():
        constructed.append(True)
        return SeestarQueuedStackerBackend(
            stacker_factory=lambda **kw: _FakeLiveStacker(**kw)
        )

    win = MainWindow(backend_factory=factory)
    try:
        assert win.is_running is False
        assert win.preview_res_button.text() == "Res 1/1"
        assert win._preview_res_factor == 1

        win.preview_res_button.click()
        assert win._preview_res_factor == 2
        assert win.preview_res_button.text() == "Res 1/2"

        # Idle: no run was started, so no backend was constructed and no worker
        # exists — the control channel was never invoked.
        assert constructed == []
        assert win.controller._worker is None
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (3) active run: Res click applies the factor to the live stacker instance
# --------------------------------------------------------------------------
def test_active_run_res_click_applies_factor_to_live_stackers(qapp):
    instances = []

    def factory(**kwargs):
        stacker = _FakeLiveStacker(**kwargs)
        instances.append(stacker)
        return stacker

    backend = SeestarQueuedStackerBackend(stacker_factory=factory, poll_interval=0.001)
    win = MainWindow(backend_factory=lambda: backend)
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: len(instances) == 1 and win.is_running)
        stacker = instances[0]

        # Qt default factor is 1; first click cycles to 2 (engine default was 2).
        win.preview_res_button.click()
        assert win._preview_res_factor == 2
        # Wait on the *call list* (the fake's default factor is already 2, so
        # ``preview_downsample_factor`` alone would not prove the control was
        # drained — the call list is the authoritative signal).
        assert _pump_until(qapp, lambda: stacker.downsample_calls == [2])
        assert stacker.preview_downsample_factor == 2
        assert stacker.downsample_calls == [2]
        assert stacker.refresh_calls >= 1
        assert win.is_running is True
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (4) no-op when no backend/stacker reachable (no crash, no error)
# --------------------------------------------------------------------------
def test_noop_when_no_backend_or_stackers_reachable():
    # Simulated backend (default): no engine -> silent no-op.
    SimulatedRunBackend().set_preview_downsample_factor(2)

    # Real backend before any run: no stacker yet -> enqueue only, no crash.
    SeestarQueuedStackerBackend().set_preview_downsample_factor(2)

    # Idle controller: no worker -> no-op.
    RunController().set_preview_downsample_factor(2)


def test_noop_when_backend_missing_method(qapp):
    """A backend that lacks the control method must not break the worker path."""
    from seestar.gui_qt.backend_runner import BackendRunResult, BaseRunBackend

    class BareBackend(BaseRunBackend):
        def run(self, request, progress_callback, log_callback,
                is_cancel_requested, preview_callback=None):
            return BackendRunResult.FINISHED

        def cancel(self):
            pass

    win = MainWindow(backend_factory=lambda: BareBackend())
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running)
        # BareBackend inherits the no-op set_preview_downsample_factor.
        win.preview_res_button.click()
        assert win._preview_res_factor == 2
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# offscreen smoke: default simulated mode (no live engine)
# --------------------------------------------------------------------------
def test_offscreen_smoke_simulated_mode_res_click_no_crash(qapp):
    """Literal ``backend_mode="simulated"`` smoke: Res never crashes a run.

    The default simulated backend has no live engine, so a Res click during the
    run is a silent no-op on the engine side (the label still cycles); after
    stop the button returns to display-only cycling.
    """
    win = MainWindow(backend_mode="simulated")
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running)

        win.preview_res_button.click()  # -> 2 (no engine, no crash)
        assert win._preview_res_factor == 2
        assert win.preview_res_button.text() == "Res 1/2"

        win.stop_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        # Idle again: cycle the label only.
        win.preview_res_button.click()  # -> 3
        assert win.preview_res_button.text() == "Res 1/3"
        assert win._preview_res_factor == 3
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (5) thread-safety smoke: GUI-thread clicks while the worker thread runs
# --------------------------------------------------------------------------
def test_thread_safety_smoke_res_clicks_during_active_run(qapp):
    instances = []

    def factory(**kwargs):
        stacker = _FakeLiveStacker(**kwargs)
        instances.append(stacker)
        return stacker

    backend = SeestarQueuedStackerBackend(stacker_factory=factory, poll_interval=0.001)
    win = MainWindow(backend_factory=lambda: backend)
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: len(instances) == 1)
        stacker = instances[0]

        # Click Res four times from the GUI thread while the worker thread keeps
        # the fake stacker running.  Each click enqueues a control request; the
        # worker drains them in FIFO order and applies them to the stacker.
        for _ in range(4):  # 1 -> 2 -> 3 -> 4 -> 1
            win.preview_res_button.click()
            qapp.processEvents()

        assert _pump_until(qapp, lambda: stacker.preview_downsample_factor == 1)
        assert stacker.downsample_calls == [2, 3, 4, 1]
        assert stacker.refresh_calls >= 4
        # No race crash: the run is still alive and the UI is consistent.
        assert win.is_running is True
        assert win.preview_res_button.text() == "Res 1/1"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (6) display-only regressions (M17/M18 invariants preserved)
# --------------------------------------------------------------------------
def test_res_click_never_mutates_preview_source(window):
    assert window._preview_source is None
    for _ in range(4):
        window.preview_res_button.click()
    assert window._preview_source is None


def test_res_click_preserves_preview_source_identity(window):
    window._on_preview(
        BackendPreviewPayload(data=_rgb(32, 16, 90, 90, 90), stack_name="res")
    )
    source = window._preview_source
    assert source is not None and not source.isNull()

    for _ in range(4):
        window.preview_res_button.click()

    assert window._preview_source is source
    assert window._preview_source is not None


def test_res_factor_still_not_in_backend_kwargs(window):
    window.preview_res_button.click()
    kw = window.build_run_request().backend_kwargs
    assert "preview_downsample_factor" not in kw
    assert "preview_res_factor" not in kw
