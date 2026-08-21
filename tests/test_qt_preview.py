"""M7/M8 seam tests: Qt preview callbacks and display-only rendering.

These tests exercise the preview seam under the ``offscreen`` Qt platform
plugin, with **no** real stacking:

* a fake backend emits a :class:`BackendPreviewPayload` that the
  :class:`RunController.preview_updated` signal and the :class:`MainWindow`
  preview label receive on the GUI thread,
* the default simulated backend emits no preview (least-disruptive path),
* :class:`SeestarQueuedStackerBackend` installs a fake stacker's
  ``set_preview_callback`` and maps the real 7-arg and 6-arg callback
  signatures (plus extra/missing args) to :class:`BackendPreviewPayload`
  without raising,
* late preview updates are suppressed after controller shutdown,
* M8 converts small image-like payloads to Qt pixmaps on the GUI thread,
* fresh-process import hygiene and source-token cleanliness hold.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QThread
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import (
    BackendPreviewPayload,
    MainWindow,
    RunController,
    RunStatus,
    create_application,
)
from seestar.gui_qt.backend_runner import (
    BackendRunResult,
    BaseRunBackend,
    SeestarQueuedStackerBackend,
)
from seestar.gui_qt.run_bridge import build_run_request
from seestar.gui_qt.preview_render import render_preview_image
from seestar.gui_qt.settings_state import QtSettingsState


def _pump_until(qapp: QApplication, predicate, timeout_ms: int = 5000) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return bool(predicate())


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _make_request(**overrides):
    state = QtSettingsState()
    for key, value in overrides.items():
        setattr(state, key, value)
    return build_run_request(state)


# --------------------------------------------------------------------------
# Fakes (test-controlled, no real engine)
# --------------------------------------------------------------------------
class PreviewEmittingBackend(BaseRunBackend):
    """Fake backend that emits one preview payload then finishes."""

    def __init__(self) -> None:
        self.run_calls = []
        self.cancel_called = False

    def run(
        self,
        request,
        progress_callback,
        log_callback,
        is_cancel_requested,
        preview_callback=None,
    ):
        self.run_calls.append(request)
        progress_callback(10)
        if preview_callback is not None:
            preview_callback(
                BackendPreviewPayload(
                    data="FAKE_DATA",
                    header={"PREV": "fake"},
                    stack_name="Stack (3/10 Img)",
                    image_count=3,
                    total_images=10,
                    current_batch=1,
                    total_batches=4,
                )
            )
        progress_callback(100)
        return BackendRunResult.FINISHED

    def cancel(self) -> None:
        self.cancel_called = True


class BlockingPreviewBackend(BaseRunBackend):
    """Fake backend that emits a preview, then blocks until cancellation."""

    def __init__(self) -> None:
        self._cancel = False
        self.cancel_called = False

    def run(
        self,
        request,
        progress_callback,
        log_callback,
        is_cancel_requested,
        preview_callback=None,
    ):
        if preview_callback is not None:
            preview_callback(BackendPreviewPayload(stack_name="queued-preview"))
        while not (is_cancel_requested() or self._cancel):
            time.sleep(0.001)
        return BackendRunResult.CANCELLED

    def cancel(self) -> None:
        self._cancel = True
        self.cancel_called = True


class PreviewStacker:
    """A fake ``SeestarQueuedStacker``-shaped object with a preview callback."""

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = dict(kwargs)
        self.align_on_disk = None
        self.progress_cb = None
        self.preview_cb = None
        self.start_kwargs = None
        self._running = False

    def set_progress_callback(self, cb) -> None:
        self.progress_cb = cb

    def set_preview_callback(self, cb) -> None:
        self.preview_cb = cb

    def start_processing(self, **kwargs):
        self.start_kwargs = dict(kwargs)
        self._running = True
        return True

    def is_running(self) -> bool:
        self._running = False
        return False

    def stop(self) -> None:
        self._running = False


class NoPreviewSetterStacker:
    """A stacker-shaped fake that genuinely lacks ``set_preview_callback``."""

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = dict(kwargs)
        self.align_on_disk = None
        self.progress_cb = None
        self.start_kwargs = None
        self._running = False

    def set_progress_callback(self, cb) -> None:
        self.progress_cb = cb

    def start_processing(self, **kwargs):
        self.start_kwargs = dict(kwargs)
        self._running = True
        return True

    def is_running(self) -> bool:
        self._running = False
        return False

    def stop(self) -> None:
        self._running = False


def _make_preview_stackers_factory(instances, *, stacker_cls=PreviewStacker):
    def factory(**kwargs):
        stacker = stacker_cls(**kwargs)
        instances.append(stacker)
        return stacker

    return factory


# --------------------------------------------------------------------------
# Controller relays a preview payload on the GUI thread
# --------------------------------------------------------------------------
def test_controller_relays_preview_payload_on_gui_thread(qapp):
    backend = PreviewEmittingBackend()
    controller = RunController()
    received = []
    finished = []

    def on_preview(payload):
        received.append((payload, QThread.currentThread()))

    controller.preview_updated.connect(on_preview)
    controller.finished.connect(lambda: finished.append(True))
    try:
        controller.start(_make_request(), backend=backend)
        assert _pump_until(qapp, lambda: controller.status is RunStatus.FINISHED)

        assert finished == [True]
        assert len(received) == 1
        payload, thread = received[0]
        assert isinstance(payload, BackendPreviewPayload)
        assert payload.data == "FAKE_DATA"
        assert payload.header == {"PREV": "fake"}
        assert payload.stack_name == "Stack (3/10 Img)"
        assert payload.image_count == 3
        assert payload.total_images == 10
        assert payload.current_batch == 1
        assert payload.total_batches == 4
        # Delivered on the GUI thread (the slot ran during processEvents).
        assert thread is qapp.thread()
        assert thread is QCoreApplication.instance().thread()
    finally:
        controller.shutdown()


def test_main_window_preview_label_updates(qapp):
    win = MainWindow(backend_factory=PreviewEmittingBackend)
    try:
        assert "Preview" in win.preview_label.text()
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        text = win.preview_label.text()
        assert "Preview: Stack (3/10 Img)" in text
        assert "3 img" in text
        assert "10" in text
        assert "batch 1" in text
        assert "4" in text
        assert win.controller.status is RunStatus.FINISHED
    finally:
        win.shutdown()


def test_simulated_backend_emits_no_preview(qapp):
    """Default simulated backend is the least-disruptive path: no previews."""
    controller = RunController()
    previews = []
    controller.preview_updated.connect(previews.append)
    try:
        controller.start(_make_request(batch_size=4), steps=5, step_delay_ms=1)
        assert _pump_until(qapp, lambda: controller.status is RunStatus.FINISHED)
        assert previews == []
    finally:
        controller.shutdown()


# --------------------------------------------------------------------------
# Late preview suppression after shutdown
# --------------------------------------------------------------------------
def test_shutdown_suppresses_queued_preview(qapp):
    """A preview emitted before shutdown must be dropped, not relayed.

    The worker emits the preview on its thread; the queued signal is only
    delivered when the GUI thread processes events.  ``shutdown()`` flips the
    controller to IDLE *before* draining the queue, so the queued preview hits
    ``_on_worker_preview`` while not RUNNING and is dropped.
    """
    backend = BlockingPreviewBackend()
    controller = RunController()
    previews = []
    controller.preview_updated.connect(previews.append)

    controller.start(_make_request(), backend=backend)
    # Give the worker thread time to emit its preview (queued, not delivered).
    time.sleep(0.05)
    controller.shutdown()
    # Drain the GUI event queue: the queued preview must be dropped.
    qapp.processEvents()
    qapp.processEvents()

    assert previews == []
    assert controller.status is RunStatus.IDLE
    assert not controller.has_live_thread


def test_controller_drops_late_preview_when_not_running(qapp):
    """Directly: a preview relayed while not RUNNING is dropped."""
    controller = RunController()
    previews = []
    controller.preview_updated.connect(previews.append)
    try:
        # No active run (status IDLE) — e.g. after shutdown.
        controller._on_worker_preview(BackendPreviewPayload(stack_name="late"))
        assert previews == []
    finally:
        controller.shutdown()


# --------------------------------------------------------------------------
# SeestarQueuedStackerBackend preview mapping (fake stacker, no engine)
# --------------------------------------------------------------------------
def _run_with_preview_backend(stacker_cls=PreviewStacker):
    instances = []
    backend = SeestarQueuedStackerBackend(
        stacker_factory=_make_preview_stackers_factory(
            instances, stacker_cls=stacker_cls
        ),
        poll_interval=0.001,
    )
    payloads = []
    result = backend.run(
        _make_request(batch_size=4),
        lambda p: None,
        lambda m: None,
        lambda: False,
        preview_callback=payloads.append,
    )
    return result, instances[0], payloads


def test_seestar_backend_maps_7_arg_preview_callback():
    result, stacker, payloads = _run_with_preview_backend()

    assert result is BackendRunResult.FINISHED
    assert callable(stacker.preview_cb)
    stacker.preview_cb("DATA", {"HDR": 1}, "Stack (3/10 Img)", 3, 10, 1, 4)

    assert len(payloads) == 1
    p = payloads[0]
    assert isinstance(p, BackendPreviewPayload)
    assert p.data == "DATA"
    assert p.header == {"HDR": 1}
    assert p.stack_name == "Stack (3/10 Img)"
    assert p.image_count == 3
    assert p.total_images == 10
    assert p.current_batch == 1
    assert p.total_batches == 4
    assert p.extra == ()


def test_seestar_backend_maps_6_arg_preview_callback():
    result, stacker, payloads = _run_with_preview_backend()

    assert result is BackendRunResult.FINISHED
    stacker.preview_cb("DATA", None, "Stack (5/20)", 5, 20, 2)

    assert len(payloads) == 1
    p = payloads[0]
    assert p.data == "DATA"
    assert p.header is None
    assert p.stack_name == "Stack (5/20)"
    assert p.image_count == 5
    assert p.total_images == 20
    assert p.current_batch == 2
    assert p.total_batches is None
    assert p.extra == ()


def test_seestar_backend_preview_tolerates_extra_and_missing_args():
    result, stacker, payloads = _run_with_preview_backend()

    assert result is BackendRunResult.FINISHED
    # Extra args beyond the known seven are preserved in ``extra``.
    stacker.preview_cb("D", "H", "name", 1, 2, 3, 4, "EXTRA1", "EXTRA2")
    assert len(payloads) == 1
    p = payloads[0]
    assert p.extra == ("EXTRA1", "EXTRA2")

    # A minimal (even malformed) call must not raise.
    payloads.clear()
    stacker.preview_cb("only-data")
    assert len(payloads) == 1
    q = payloads[0]
    assert q.data == "only-data"
    assert q.header is None
    assert q.stack_name == ""
    assert q.image_count is None


def test_seestar_backend_without_preview_setter_is_tolerated():
    """A stacker without ``set_preview_callback`` must not crash the run."""
    result, stacker, payloads = _run_with_preview_backend(NoPreviewSetterStacker)

    assert result is BackendRunResult.FINISHED
    # The adapter is never installed because the stacker has no setter.
    assert not hasattr(stacker, "preview_cb")
    assert payloads == []


# --------------------------------------------------------------------------
# M8: display-only image rendering (Qt pixmap from BackendPreviewPayload.data)
# --------------------------------------------------------------------------
class ImagePreviewBackend(BaseRunBackend):
    """Fake backend that emits one preview payload with arbitrary ``data``."""

    def __init__(self, data, stack_name="img-preview") -> None:
        self._data = data
        self._stack_name = stack_name
        self.cancel_called = False

    def run(
        self,
        request,
        progress_callback,
        log_callback,
        is_cancel_requested,
        preview_callback=None,
    ):
        progress_callback(50)
        if preview_callback is not None:
            preview_callback(
                BackendPreviewPayload(
                    data=self._data,
                    stack_name=self._stack_name,
                    image_count=1,
                    total_images=5,
                )
            )
        progress_callback(100)
        return BackendRunResult.FINISHED

    def cancel(self) -> None:
        self.cancel_called = True


def test_renderer_returns_copied_qimage_for_uint8_rgb():
    arr = np.zeros((3, 5, 3), dtype=np.uint8)
    arr[:, :, 0] = 255
    img = render_preview_image(arr)
    assert img is not None
    assert not img.isNull()
    assert img.width() == 5
    assert img.height() == 3


def test_renderer_tolerates_invalid_data():
    assert render_preview_image(None) is None
    assert render_preview_image("not-an-image") is None
    assert render_preview_image(12345) is None
    assert render_preview_image([]) is None
    assert render_preview_image(np.array([1, 2, 3])) is None  # 1D -> not an image
    assert render_preview_image(np.zeros((0, 4, 3), dtype=np.uint8)) is None
    assert render_preview_image(np.zeros((4, 4, 5), dtype=np.uint8)) is None  # 5 ch


def test_main_window_preview_pixmap_from_float_rgb(qapp):
    arr = np.zeros((8, 12, 3), dtype=np.float32)
    arr[:, :, 0] = 1.0  # pure red
    win = MainWindow(backend_factory=lambda: ImagePreviewBackend(arr, "rgb-preview"))
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        assert "rgb-preview" in win.preview_label.text()
        pixmap = win.preview_image_label.pixmap()
        assert pixmap is not None
        assert not pixmap.isNull()
        assert pixmap.width() == 12
        assert pixmap.height() == 8
        color = pixmap.toImage().pixelColor(0, 0)
        assert color.red() > 200
        assert color.green() < 20
        assert color.blue() < 20
    finally:
        win.shutdown()


def test_main_window_preview_pixmap_from_2d_mono(qapp):
    arr = np.full((6, 10), 0.5, dtype=np.float32)
    win = MainWindow(backend_factory=lambda: ImagePreviewBackend(arr, "mono-preview"))
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        pixmap = win.preview_image_label.pixmap()
        assert not pixmap.isNull()
        assert pixmap.width() == 10
        assert pixmap.height() == 6
        color = pixmap.toImage().pixelColor(0, 0)
        assert color.red() == color.green() == color.blue()
        assert abs(color.red() - 255) <= 2
    finally:
        win.shutdown()


def test_main_window_preview_pixmap_from_tuple_payload(qapp):
    display = np.zeros((5, 7, 3), dtype=np.float32)
    display[:, :, 1] = 1.0  # pure green
    hist = np.array([0.1, 0.2, 0.3])  # 1D, not an image
    win = MainWindow(
        backend_factory=lambda: ImagePreviewBackend((display, hist), "tuple-preview")
    )
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        pixmap = win.preview_image_label.pixmap()
        assert not pixmap.isNull()
        assert pixmap.width() == 7
        assert pixmap.height() == 5
        color = pixmap.toImage().pixelColor(0, 0)
        assert color.green() > 200
        assert color.red() < 20
        assert color.blue() < 20
    finally:
        win.shutdown()


def test_renderer_tuple_payload_skips_non_image_array_candidate():
    """Tuple/list payloads render the first image-like candidate, not just any array."""
    hist = np.array([0.1, 0.2, 0.3])  # array-like but not image-like
    display = np.zeros((4, 6, 3), dtype=np.float32)
    display[:, :, 1] = 1.0

    img = render_preview_image((hist, display))

    assert img is not None
    assert not img.isNull()
    assert img.width() == 6
    assert img.height() == 4
    color = img.pixelColor(0, 0)
    assert color.green() > 200
    assert color.red() < 20
    assert color.blue() < 20


def test_main_window_preview_invalid_data_clears_pixmap(qapp):
    """Invalid data never crashes and clears the image area (documented policy)."""
    win = MainWindow()
    try:
        arr = np.zeros((4, 4, 3), dtype=np.float32)
        arr[:, :, 2] = 1.0
        win._on_preview(BackendPreviewPayload(data=arr, stack_name="ok"))
        assert not win.preview_image_label.pixmap().isNull()
        assert "ok" in win.preview_label.text()

        # Invalid data: metadata label still updates, image area clears.
        win._on_preview(BackendPreviewPayload(data="garbage", stack_name="bad"))
        assert win.preview_image_label.pixmap().isNull()
        assert "bad" in win.preview_label.text()

        # Missing data also clears without raising.
        win._on_preview(BackendPreviewPayload(data=None, stack_name="none"))
        assert win.preview_image_label.pixmap().isNull()
        assert "none" in win.preview_label.text()
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Source-token / import-hygiene invariants for the new seam
# --------------------------------------------------------------------------
def test_preview_source_is_tk_engine_numpy_free():
    from pathlib import Path

    pkg_dir = Path(__file__).resolve().parents[1] / "seestar" / "gui_qt"
    forbidden = (
        "seestar.core",
        "seestar.alignment",
        "seestar.enhancement",
        "seestar.queuep",
        "tkinter",
        "seestar.gui.settings",
        "seestar.gui.main_window",
        "seestar.gui.boring_stack",
        "zesolver_adapter",
        "zesolver.api",
        "zealfie",
        "PIL",
        "matplotlib",
    )
    # numpy is allowed as a *lazy* import inside the renderer; its absence at
    # import time is asserted by test_preview_import_hygiene_fresh_process below
    # (via sys.modules), not by forbidding the source token here.
    for py in sorted(pkg_dir.glob("*.py")):
        text = py.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{py.name} references {token}"


def test_preview_import_hygiene_fresh_process():
    """Fresh interpreter: importing the package must stay Tk/engine clean."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "import seestar.gui_qt  # noqa: F401\n"
        "from seestar.gui_qt import BackendPreviewPayload\n"
        "from seestar.gui_qt.backend_runner import BackendPreviewPayload as B2\n"
        "assert BackendPreviewPayload is B2\n"
        "_bad = [m for m in sys.modules\n"
        "        if m.startswith('tkinter')\n"
        "        or m.startswith('seestar.core')\n"
        "        or m.startswith('seestar.alignment')\n"
        "        or m.startswith('seestar.enhancement')\n"
        "        or m.startswith('seestar.queuep')\n"
        "        or m in ('seestar.gui.main_window', 'seestar.gui.settings',"
        " 'seestar.gui.boring_stack')\n"
        "        or m.split('.')[0] in ('numpy', 'PIL', 'matplotlib')]\n"
        "if _bad:\n"
        "    print('BAD_MODULES:', _bad)\n"
        "    sys.exit(1)\n"
        "from seestar.gui_qt.run_bridge import RunRequest as Q\n"
        "from seestar.gui.run_config import RunRequest as C\n"
        "assert Q is C\n"
        "print('IMPORT_HYGIENE_OK')\n"
    )
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
    )
    assert proc.returncode == 0, (
        f"preview import hygiene violated: stdout={proc.stdout!r} "
        f"stderr={proc.stderr!r}"
    )


# --------------------------------------------------------------------------
# M5: preview view controls (zoom / resolution / rotation) — display-only
# --------------------------------------------------------------------------
def _preview_uint8(width: int, height: int):
    """Build a ``width x height`` RGB uint8 array (width=W, height=H)."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_view_controls_initially_disabled_and_reset(qapp):
    win = MainWindow()
    try:
        assert not win.zoom_combo.isEnabled()
        assert not win.rotate_left_button.isEnabled()
        assert not win.rotate_right_button.isEnabled()
        assert win.resolution_label.text() == "—"
        assert win.has_preview_image is False
        assert win.preview_rotation == 0
        assert win.zoom_combo.currentText() == "100%"
    finally:
        win.shutdown()


def test_image_preview_enables_view_controls_and_sets_resolution(qapp):
    win = MainWindow(backend_factory=lambda: ImagePreviewBackend(_preview_uint8(20, 10)))
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        # After a finished run the view controls stay usable (image retained).
        assert win.has_preview_image is True
        assert win.zoom_combo.isEnabled()
        assert win.rotate_left_button.isEnabled()
        assert win.rotate_right_button.isEnabled()
        assert win.zoom_combo.currentText() == "100%"
        assert win.resolution_label.text() == "20x10 → 20x10 · 100%"
        assert win.preview_image_label.pixmap().width() == 20
        assert win.preview_image_label.pixmap().height() == 10
    finally:
        win.shutdown()


def test_zoom_percent_levels_produce_expected_dimensions(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="z"))

        win.zoom_combo.setCurrentText("100%")
        assert win.preview_image_label.pixmap().width() == 20
        assert win.preview_image_label.pixmap().height() == 10
        assert win.resolution_label.text() == "20x10 → 20x10 · 100%"

        win.zoom_combo.setCurrentText("200%")
        assert win.preview_image_label.pixmap().width() == 40
        assert win.preview_image_label.pixmap().height() == 20
        assert win.resolution_label.text() == "20x10 → 40x20 · 200%"

        win.zoom_combo.setCurrentText("50%")
        assert win.preview_image_label.pixmap().width() == 10
        assert win.preview_image_label.pixmap().height() == 5
        assert win.resolution_label.text() == "20x10 → 10x5 · 50%"
    finally:
        win.shutdown()


def test_zoom_fit_preserves_aspect_ratio(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="f"))

        # Square label (>= the 256px minimum): 20x10 (2:1) fits into 400x400
        # width-limited -> 400x200.
        win.preview_image_label.resize(400, 400)
        win.zoom_combo.setCurrentText("Fit")
        pixmap = win.preview_image_label.pixmap()
        assert pixmap.width() == 400
        assert pixmap.height() == 200
        assert win.resolution_label.text() == "20x10 → 400x200 · Fit"

        # Height-limited label re-fits against the new target, still 2:1.
        win.preview_image_label.resize(1000, 300)
        win._refresh_preview_view()
        pixmap = win.preview_image_label.pixmap()
        assert pixmap.width() == 600
        assert pixmap.height() == 300
        assert win.resolution_label.text() == "20x10 → 600x300 · Fit"
    finally:
        win.shutdown()


def test_rotation_swaps_dimensions_and_is_cumulative(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="r"))
        win.zoom_combo.setCurrentText("100%")

        win.rotate_right_button.click()
        assert win.preview_rotation == 90
        assert win.preview_image_label.pixmap().width() == 10
        assert win.preview_image_label.pixmap().height() == 20
        assert "90°" in win.resolution_label.text()

        win.rotate_right_button.click()
        assert win.preview_rotation == 180
        assert win.preview_image_label.pixmap().width() == 20
        assert win.preview_image_label.pixmap().height() == 10

        win.rotate_left_button.click()
        assert win.preview_rotation == 90
        assert win.preview_image_label.pixmap().width() == 10
        assert win.preview_image_label.pixmap().height() == 20

        # Full 360 accumulation wraps back to 0 and the native orientation.
        win.rotate_right_button.click()  # 180
        win.rotate_right_button.click()  # 270
        win.rotate_right_button.click()  # 0
        assert win.preview_rotation == 0
        assert win.preview_image_label.pixmap().width() == 20
        assert win.preview_image_label.pixmap().height() == 10
        assert "90°" not in win.resolution_label.text()
    finally:
        win.shutdown()


def test_rotation_resets_on_new_image_and_clear(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="a"))
        win.rotate_right_button.click()
        assert win.preview_rotation == 90

        # A new image starts unrotated and re-renders at the current zoom.
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(8, 12), stack_name="b"))
        assert win.preview_rotation == 0
        assert win.preview_image_label.pixmap().width() == 8
        assert win.preview_image_label.pixmap().height() == 12

        # Clearing the image resets rotation and disables the controls.
        win._on_preview(BackendPreviewPayload(data=None, stack_name="none"))
        assert win.preview_rotation == 0
        assert win.has_preview_image is False
    finally:
        win.shutdown()


def test_non_image_payload_clears_and_disables_view_controls(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="ok"))
        assert win.has_preview_image is True
        assert win.zoom_combo.isEnabled()
        assert win.rotate_left_button.isEnabled()
        assert win.rotate_right_button.isEnabled()

        win._on_preview(BackendPreviewPayload(data="garbage", stack_name="bad"))
        assert win.has_preview_image is False
        assert win.preview_image_label.pixmap().isNull()
        assert not win.zoom_combo.isEnabled()
        assert not win.rotate_left_button.isEnabled()
        assert not win.rotate_right_button.isEnabled()
        assert win.resolution_label.text() == "—"
        assert win.preview_rotation == 0
        # Metadata label still reflects the (invalid-data) payload.
        assert "bad" in win.preview_label.text()
    finally:
        win.shutdown()


def test_view_controls_do_not_alter_metadata_label(qapp):
    win = MainWindow(backend_factory=lambda: ImagePreviewBackend(_preview_uint8(20, 10), "meta"))
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)
        before = win.preview_label.text()
        assert "meta" in before
        assert "1 img" in before

        win.zoom_combo.setCurrentText("200%")
        win.rotate_left_button.click()
        win.rotate_right_button.click()
        win.zoom_combo.setCurrentText("Fit")

        assert win.preview_label.text() == before
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# M10/M14: preview controls (WB / stretch) — display-only.  The duplicated tab
# histogram was removed in M14; the right-panel histogram is the single surface.
# --------------------------------------------------------------------------
def test_preview_controls_tab_has_wb_stretch_and_no_histogram(qapp):
    win = MainWindow()
    try:
        for attr in (
            "wb_group",
            "wb_r_spin",
            "wb_g_spin",
            "wb_b_spin",
            "wb_reset_button",
            "stretch_group",
            "stretch_combo",
            "auto_stretch_button",
        ):
            assert hasattr(win, attr), f"missing preview-control {attr}"
        # M14: the duplicated tab histogram is removed (not merely hidden), so
        # the tab no longer owns any histogram widgets.
        for removed in ("histogram_group", "histogram_status", "histogram_view"):
            assert not hasattr(win, removed), f"tab histogram {removed} should be gone"
        assert [
            win.stretch_combo.itemText(i)
            for i in range(win.stretch_combo.count())
        ] == ["linear", "asinh", "log", "auto"]
        assert win.stretch_combo.currentText() == "asinh"
        assert (
            win.wb_r_spin.value(),
            win.wb_g_spin.value(),
            win.wb_b_spin.value(),
        ) == (1.0, 1.0, 1.0)
        # Inert before a renderable preview exists.
        assert not win.wb_r_spin.isEnabled()
        assert not win.wb_reset_button.isEnabled()
        assert not win.stretch_combo.isEnabled()
        assert not win.auto_stretch_button.isEnabled()
        assert not win.right_histogram_view.has_data
    finally:
        win.shutdown()


def test_preview_controls_enable_with_renderable_preview(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="pc"))
        assert win.wb_r_spin.isEnabled()
        assert win.wb_reset_button.isEnabled()
        assert win.stretch_combo.isEnabled()
        assert win.auto_stretch_button.isEnabled()
        assert win.right_histogram_view.has_data
        # Clearing disables them again.
        win._on_preview(BackendPreviewPayload(data=None, stack_name="none"))
        assert not win.wb_r_spin.isEnabled()
        assert not win.stretch_combo.isEnabled()
        assert not win.right_histogram_view.has_data
    finally:
        win.shutdown()


def test_neutral_wb_stretch_keeps_dimensions_zoom_rotation(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="n"))
        # Neutral defaults: byte-identical to the pre-M10 render (M5 behaviour).
        assert win.preview_image_label.pixmap().width() == 20
        assert win.preview_image_label.pixmap().height() == 10
        win.zoom_combo.setCurrentText("200%")
        assert win.preview_image_label.pixmap().width() == 40
        assert win.preview_image_label.pixmap().height() == 20
        win.rotate_right_button.click()
        assert win.preview_rotation == 90
        assert win.preview_image_label.pixmap().width() == 20
        assert win.preview_image_label.pixmap().height() == 40
        assert win.resolution_label.text() == "20x10 → 20x40 · 200% · 90°"
    finally:
        win.shutdown()


def test_wb_change_alters_displayed_pixmap_deterministically(qapp):
    win = MainWindow()
    try:
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        arr[:, :, 0] = 200  # pure red
        win._on_preview(BackendPreviewPayload(data=arr, stack_name="wb"))
        win.stretch_bp_spin.setValue(0.0)
        win.stretch_wp_spin.setValue(1.0)
        win.stretch_combo.setCurrentText("linear")

        before = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)
        assert before.red() == 200
        assert before.green() == 0
        assert before.blue() == 0
        source_before = win._preview_source
        assert source_before is not None

        win.wb_r_spin.setValue(0.5)  # halve the red gain
        after = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)
        assert abs(after.red() - 100) <= 1
        assert after.green() == 0
        assert after.blue() == 0

        # The stored source image is untouched by the WB control.
        assert win._preview_source is source_before
        assert win._preview_source.pixelColor(0, 0).red() == 200

        win.wb_reset_button.click()
        reset = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)
        assert reset.red() == 200
    finally:
        win.shutdown()


def test_stretch_change_alters_displayed_pixmap_deterministically(qapp):
    win = MainWindow()
    try:
        grad = np.tile(np.array([0, 64, 128, 255], dtype=np.uint8), (2, 1))
        win._on_preview(BackendPreviewPayload(data=grad, stack_name="stretch"))

        # Isolate the stretch curve: identity black/white + unit gamma.
        win.stretch_bp_spin.setValue(0.0)
        win.stretch_wp_spin.setValue(1.0)
        win.stretch_combo.setCurrentText("linear")
        lin = win.preview_image_label.pixmap().toImage().pixelColor(2, 0)
        assert lin.red() == 128  # linear identity leaves the mid-tone unchanged

        win.stretch_combo.setCurrentText("log")
        logp = win.preview_image_label.pixmap().toImage().pixelColor(2, 0)
        assert logp.red() != lin.red()

        win.stretch_combo.setCurrentText("asinh")
        asinhp = win.preview_image_label.pixmap().toImage().pixelColor(2, 0)
        assert asinhp.red() != lin.red()
        assert asinhp.red() != logp.red()
    finally:
        win.shutdown()


def test_auto_stretch_expands_low_contrast_gradient(qapp):
    win = MainWindow()
    try:
        grad = np.tile(np.array([50, 60, 70, 80], dtype=np.uint8), (2, 1))
        win._on_preview(BackendPreviewPayload(data=grad, stack_name="auto"))

        # Identity stretch first, then "auto" (min/max normalisation).
        win.stretch_bp_spin.setValue(0.0)
        win.stretch_wp_spin.setValue(1.0)
        win.stretch_combo.setCurrentText("linear")
        assert win.preview_image_label.pixmap().toImage().pixelColor(0, 0).red() == 50

        win.stretch_combo.setCurrentText("auto")
        assert win.preview_image_label.pixmap().toImage().pixelColor(0, 0).red() == 0
        assert win.preview_image_label.pixmap().toImage().pixelColor(3, 0).red() == 255
    finally:
        win.shutdown()


def test_histogram_updates_and_clears(qapp):
    win = MainWindow()
    try:
        assert not win.right_histogram_view.has_data
        assert win._histogram_stats is None

        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[:, :, 0] = 200
        win._on_preview(BackendPreviewPayload(data=arr, stack_name="h"))
        assert win.right_histogram_view.has_data
        assert win._histogram_stats is not None
        assert "R" in win._histogram_stats
        assert "200" in win._histogram_stats

        # Non-image preview clears the histogram without crashing.
        win._on_preview(BackendPreviewPayload(data="garbage", stack_name="bad"))
        assert not win.right_histogram_view.has_data
        assert win._histogram_stats is None
    finally:
        win.shutdown()


def test_histogram_updates_when_controls_change(qapp):
    win = MainWindow()
    try:
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[:, :, 0] = 200
        win._on_preview(BackendPreviewPayload(data=arr, stack_name="hc"))
        win.stretch_bp_spin.setValue(0.0)
        win.stretch_wp_spin.setValue(1.0)
        win.stretch_combo.setCurrentText("linear")
        before = win._histogram_stats
        assert before is not None
        assert "R 200" in before

        win.wb_r_spin.setValue(0.5)  # attenuate red -> stats change
        after = win._histogram_stats
        assert after is not None
        assert after != before
        assert "R 100" in after
    finally:
        win.shutdown()


def test_right_panel_is_single_histogram_surface(qapp):
    """M14: the right-panel histogram is the single live surface (no duplicate)."""
    win = MainWindow()
    try:
        # The former tab surface no longer exists.
        assert not hasattr(win, "histogram_view")
        assert not hasattr(win, "histogram_status")
        assert not hasattr(win, "histogram_group")
        assert not win.right_histogram_view.has_data
        assert win.right_histogram_status.text() == "No preview"

        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[:, :, 0] = 200
        win._on_preview(BackendPreviewPayload(data=arr, stack_name="rh"))

        assert win.right_histogram_view.has_data
        assert win._histogram_stats is not None
        assert "Stats:" in win.right_histogram_status.text()
        assert "200" in win.right_histogram_status.text()

        # Non-image preview clears the single surface.
        win._on_preview(BackendPreviewPayload(data="garbage", stack_name="bad"))
        assert not win.right_histogram_view.has_data
        assert win._histogram_stats is None
        assert win.right_histogram_status.text() == "No preview"
    finally:
        win.shutdown()


def test_zoom_rotation_still_apply_after_wb_stretch(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_preview_uint8(20, 10), stack_name="mix"))
        win.wb_r_spin.setValue(0.5)
        win.stretch_combo.setCurrentText("asinh")
        win.zoom_combo.setCurrentText("200%")
        win.rotate_right_button.click()

        assert win.preview_image_label.pixmap().width() == 20
        assert win.preview_image_label.pixmap().height() == 40
        assert "200%" in win.resolution_label.text()
        assert "90°" in win.resolution_label.text()
    finally:
        win.shutdown()


def test_preview_adjust_module_is_display_only():
    """The new helper keeps the same source-token / import-hygiene invariants."""
    from pathlib import Path

    import seestar.gui_qt as gui_qt

    pkg_dir = Path(gui_qt.__file__).resolve().parent
    text = (pkg_dir / "preview_adjust.py").read_text(encoding="utf-8")
    forbidden = (
        "seestar.core",
        "seestar.alignment",
        "seestar.enhancement",
        "seestar.queuep",
        "tkinter",
        "seestar.gui.settings",
        "seestar.gui.main_window",
        "seestar.gui.boring_stack",
        "zesolver_adapter",
        "zesolver.api",
        "zealfie",
        "PIL",
        "matplotlib",
    )
    for token in forbidden:
        assert token not in text, f"preview_adjust.py references {token}"
    # numpy stays a *lazy* import: no top-level ``import numpy`` statement.
    for line in text.splitlines():
        stripped = line.strip()
        assert not stripped.startswith(("import numpy", "from numpy")), (
            f"preview_adjust.py imports numpy at top level: {line!r}"
        )
