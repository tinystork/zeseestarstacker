"""M24 tests: preview display reconciliation (close the M22 caveat, 16.15 note).

The M22 lot delivered the *control* half of the Res-factor story — the Qt
``preview_res_button`` now drives the live engine ``preview_downsample_factor``
during an active run through a thread-safe control channel.  The *display* half
was left as a documented caveat: the engine pushes preview frames already
downsampled by its own factor, while the Qt shell additionally applied its own
local display downsample (``_preview_res_factor``) on top in the render path, so
the on-screen preview did not reflect the engine factor 1:1.

M24 reconciles that by making the engine factor the single source of truth for
the rendered preview **during an active run** (render the engine-pushed frames
at their native pushed resolution, no local display downsample on top), while
the idle (no run) display-only local factor behaviour from M17/M18 stays
byte-identical.  The Tk GUI and the engine are NOT modified.

Tests (engine-free, faithful fakes; ``_preview_source`` only ever updated via
the existing ``_on_preview`` flow):

1. active run + engine factor 2 -> render uses the engine-native resolution
   (no double downsample),
2. active run factor change mid-run (M22 channel) -> the next frame renders at
   the new engine factor,
3. idle -> local display factor behaviour unchanged (M17/M18 regression),
4. run end (finished/failed/cancelled) -> idle local-factor behaviour restored,
5. no crash when a payload carries no geometry / no header,
6. pan/zoom (M18) unaffected during active-run rendering.

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
from seestar.gui_qt.backend_runner import SeestarQueuedStackerBackend
from seestar.gui_qt.main_window import DEFAULT_PREVIEW_RES_FACTOR, PREVIEW_RES_FACTORS


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
    """Faithful fake exposing the live preview-downsample control surface."""

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
# (1) active run + engine factor 2 -> engine-native resolution, no double
#     downsample
# --------------------------------------------------------------------------
def test_active_run_renders_engine_native_resolution(qapp):
    win = MainWindow()
    try:
        win._on_run_started()
        assert win.is_running is True

        # The user cycles Res to 1/2 during the run (M22 also forwards this to
        # the live engine; with no worker the forward is a silent no-op).
        win.preview_res_button.click()
        assert win._preview_res_factor == 2

        # The engine then pushes a frame already downsampled 1/2 (128x64 -> 64x32).
        win._on_preview(
            BackendPreviewPayload(data=_rgb(64, 32, 0, 128, 255), stack_name="run")
        )
        pm = win.preview_image_label.pixmap()
        assert (pm.width(), pm.height()) == (64, 32)
        # The resolution label reflects the engine factor (no local downsample
        # on top -> 64x32, not 32x16).
        assert win.resolution_label.text() == "64x32 → 64x32 · 100%"
    finally:
        win.shutdown()


def test_active_run_ignores_local_factor_in_render_path(qapp):
    """During a run the render path must ignore ``_preview_res_factor`` entirely.

    Whatever the local factor is, the displayed resolution is the payload's
    native (engine-pushed) resolution — the engine factor is the single source
    of truth.
    """
    win = MainWindow()
    try:
        win._on_run_started()
        # Even an extreme local factor (4) must not downsample the engine frame.
        win._preview_res_factor = 4
        win._render_preview_res_button()

        win._on_preview(
            BackendPreviewPayload(data=_rgb(64, 32, 10, 20, 30), stack_name="run4")
        )
        pm = win.preview_image_label.pixmap()
        assert (pm.width(), pm.height()) == (64, 32)
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (2) active run factor change mid-run (M22 channel) -> next frame at the new
#     engine factor
# --------------------------------------------------------------------------
def test_active_run_factor_change_mid_run_renders_new_engine_resolution(qapp):
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

        # Engine factor 1: the engine pushes a full-resolution frame (128x64).
        win._on_preview(
            BackendPreviewPayload(data=_rgb(128, 64, 40, 40, 40), stack_name="f1")
        )
        assert (win.preview_image_label.pixmap().width(),
                win.preview_image_label.pixmap().height()) == (128, 64)

        # M22 channel: Res click -> engine factor 2, applied on the worker thread.
        win.preview_res_button.click()
        assert win._preview_res_factor == 2
        assert _pump_until(qapp, lambda: stacker.downsample_calls == [2])
        assert stacker.preview_downsample_factor == 2

        # Next frame arrives at the new engine factor (128x64 -> 64x32).
        win._on_preview(
            BackendPreviewPayload(data=_rgb(64, 32, 80, 80, 80), stack_name="f2")
        )
        pm = win.preview_image_label.pixmap()
        assert (pm.width(), pm.height()) == (64, 32)
        assert win.resolution_label.text() == "64x32 → 64x32 · 100%"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (3) idle -> local display factor behaviour unchanged (M17/M18 regression)
# --------------------------------------------------------------------------
def test_idle_local_display_factor_unchanged(qapp):
    win = MainWindow()
    try:
        assert win.is_running is False
        win._on_preview(
            BackendPreviewPayload(data=_rgb(64, 32, 0, 128, 255), stack_name="idle")
        )
        win.zoom_combo.setCurrentText("100%")
        # Idle factor 1 -> native.
        assert win.preview_image_label.pixmap().width() == 64

        win.preview_res_button.click()  # -> 2
        assert win.preview_image_label.pixmap().width() == 32
        win.preview_res_button.click()  # -> 3
        win.preview_res_button.click()  # -> 4
        assert win.preview_image_label.pixmap().width() == 16
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (4) run end -> idle local-factor behaviour restored
# --------------------------------------------------------------------------
def test_run_end_restores_idle_local_factor(qapp):
    win = MainWindow()
    try:
        assert win._effective_preview_downsample_factor() == DEFAULT_PREVIEW_RES_FACTOR == 1

        win._on_run_started()
        assert win._effective_preview_downsample_factor() == 1
        # Local factor changes while running never leak into the render path.
        win._preview_res_factor = 3
        assert win._effective_preview_downsample_factor() == 1

        # finished -> idle local factor restored.
        win._on_run_finished()
        assert win.is_running is False
        assert win._effective_preview_downsample_factor() == 3

        # failed / cancelled restore idle behaviour too (no stale engine factor).
        win._on_run_started()
        assert win._effective_preview_downsample_factor() == 1
        win._on_run_failed("boom")
        assert win.is_running is False
        assert win._effective_preview_downsample_factor() == 3

        win._on_run_started()
        win._on_run_cancelled()
        assert win.is_running is False
        assert win._effective_preview_downsample_factor() == 3
    finally:
        win.shutdown()


def test_run_end_render_returns_to_local_factor(qapp):
    win = MainWindow()
    try:
        win._on_preview(
            BackendPreviewPayload(data=_rgb(64, 32, 1, 2, 3), stack_name="src")
        )
        win.zoom_combo.setCurrentText("100%")
        assert win.preview_image_label.pixmap().width() == 64  # idle factor 1

        win._on_run_started()
        win._preview_res_factor = 2
        win._render_preview_res_button()
        win._refresh_preview_view()
        assert win.preview_image_label.pixmap().width() == 64  # run: engine-native

        win._on_run_finished()
        win._refresh_preview_view()
        assert win.preview_image_label.pixmap().width() == 32  # idle factor 2 again
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (5) no crash when a payload lacks geometry / header
# --------------------------------------------------------------------------
def test_no_crash_when_payload_lacks_geometry_and_header(qapp):
    win = MainWindow()
    try:
        win._on_run_started()
        # No data, no header, no geometry fields.
        win._on_preview(BackendPreviewPayload(data=None, header=None, stack_name="x"))
        # Non-image data (a plain string has no shape).
        win._on_preview(
            BackendPreviewPayload(data="not_an_image", header=None, stack_name="y")
        )
        # Metadata-only payload (counts, no pixel data).
        win._on_preview(
            BackendPreviewPayload(stack_name="z", image_count=3, total_images=10)
        )
        # All invalid payloads cleared the source and never raised.
        assert win._preview_source is None or win._preview_source.isNull()
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (6) pan/zoom (M18) unaffected during active-run rendering
# --------------------------------------------------------------------------
def test_pan_zoom_unaffected_during_active_run(qapp):
    win = MainWindow()
    try:
        win._on_run_started()
        win._preview_res_factor = 2  # UI factor 2 during the run
        win._render_preview_res_button()
        win._on_preview(
            BackendPreviewPayload(data=_rgb(64, 32, 7, 8, 9), stack_name="run")
        )

        # 200% zoom is layered on the engine-native frame (64x32 -> 128x64),
        # NOT on a double-downsampled frame (32x16 -> 64x32).
        win.zoom_combo.setCurrentText("200%")
        pm = win.preview_image_label.pixmap()
        assert (pm.width(), pm.height()) == (128, 64)

        # Pan accumulates the viewport offset without disturbing the source.
        win.zoom_combo.setCurrentText("100%")
        win._on_pan_delta(7, -3)
        assert win._view_offset_x == 7.0
        assert win._view_offset_y == -3.0
        assert win._preview_source is not None
    finally:
        win.shutdown()
