"""M18 seam tests: preview pan/zoom parity (checklist item 6.7).

Offscreen tests for the M18 lot:

* the pure view-transform helpers mirror the Tk ``PreviewManager`` zoom
  constants (``MIN_ZOOM`` / ``MAX_ZOOM`` / ``ZOOM_STEP``) and never mutate a
  source image,
* mouse-wheel zoom changes the continuous zoom factor, re-renders the preview
  and preserves the stored ``_preview_source`` identity (no source mutation),
* left-drag pan changes the viewport offset (unbounded — Tk clamps nothing),
* the ``zoom_combo`` presets interact with wheel zoom per the documented rule
  (presets set the factor + recentre; wheel zoom sets a continuous factor that
  may fall outside the presets and shows a blank combo),
* pan/zoom reset on a new preview image and on "Zoom Fit",
* the numeric zoom label is language-neutral (no new FR/EN keys required), and
* the engine-coupled absence assertions hold (no engine/Tk import paths).

No real stacking, no engine, no Tk.  ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import BackendPreviewPayload, MainWindow, create_application
from seestar.gui_qt.preview_image_view import PreviewImageView
from seestar.gui_qt.preview_view import (
    MAX_ZOOM,
    MIN_ZOOM,
    ZOOM_STEP,
    clamp_zoom_factor,
    compose_panned_pixmap,
    fit_scale,
    preset_label_for_factor,
    scaled_image,
    zoomed_image_size,
)


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


def _rgb(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _load(window, width=64, height=32, name="m18"):
    window._on_preview(BackendPreviewPayload(data=_rgb(width, height), stack_name=name))
    return window._preview_source


# --------------------------------------------------------------------------
# (0) pure view-transform helpers mirror Tk
# --------------------------------------------------------------------------
def test_zoom_constants_mirror_tk():
    assert MIN_ZOOM == 0.05
    assert MAX_ZOOM == 15.0
    assert ZOOM_STEP == 1.15


def test_clamp_zoom_factor():
    assert clamp_zoom_factor(0.001) == MIN_ZOOM
    assert clamp_zoom_factor(100.0) == MAX_ZOOM
    assert clamp_zoom_factor(1.5) == 1.5
    assert clamp_zoom_factor(2.0) == 2.0


def test_preset_label_for_factor():
    assert preset_label_for_factor(1.0) == "100%"
    assert preset_label_for_factor(2.0) == "200%"
    assert preset_label_for_factor(0.5) == "50%"
    assert preset_label_for_factor(1.15) is None
    assert preset_label_for_factor(15.0) is None


def test_scaled_image_never_mutates_source(qapp):
    img = QImage(64, 64, QImage.Format.Format_RGB32)
    img.fill(0xFF102030)
    out = scaled_image(img, 2.0)
    assert out is not img
    assert (out.width(), out.height()) == (128, 128)
    # factor 1.0 / non-positive return the source unchanged.
    assert scaled_image(img, 1.0) is img
    assert scaled_image(img, 0.0) is img
    assert scaled_image(img, -2.0) is img
    assert (img.width(), img.height()) == (64, 64)


def test_zoomed_image_size_matches_numeric_render(qapp):
    img = QImage(64, 32, QImage.Format.Format_RGB32)
    img.fill(0xFF000000)
    assert zoomed_image_size(img, 0, 1, 1.0) == (64, 32)
    assert zoomed_image_size(img, 0, 1, 2.0) == (128, 64)
    assert zoomed_image_size(img, 90, 1, 1.0) == (32, 64)
    assert zoomed_image_size(img, 0, 2, 1.0) == (32, 16)


def test_fit_scale_preserves_aspect(qapp):
    img = QImage(64, 32, QImage.Format.Format_RGB32)
    img.fill(0xFF000000)
    from PySide6.QtCore import QSize

    assert fit_scale(img, 0, 1, QSize(200, 200)) == pytest.approx(200 / 64)
    assert fit_scale(img, 0, 1, QSize(32, 200)) == pytest.approx(32 / 64)
    assert fit_scale(img, 0, 1, QSize(0, 0)) == 1.0


def test_compose_panned_pixmap_clips_to_viewport(qapp):
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QPixmap

    pixmap = QPixmap(10, 10)
    pixmap.fill(Qt.GlobalColor.black)
    out = compose_panned_pixmap(pixmap, QSize(50, 30), (0.0, 0.0))
    assert (out.width(), out.height()) == (50, 30)
    # Degenerate target returns the pixmap unchanged.
    assert compose_panned_pixmap(pixmap, QSize(0, 0), (0.0, 0.0)) is pixmap


# --------------------------------------------------------------------------
# (1) mouse-wheel zoom: state + re-render, no ``_preview_source`` mutation
# --------------------------------------------------------------------------
def test_wheel_zoom_changes_factor_and_rerenders_without_touching_source(window):
    source = _load(window)
    assert window._preview_zoom_factor == 1.0

    cx = window.preview_image_label.width() / 2.0
    cy = window.preview_image_label.height() / 2.0

    window._on_wheel_zoom(1, cx, cy)  # zoom in -> 1.15
    assert window._preview_zoom_factor == pytest.approx(1.15)
    assert window._preview_source is source  # identity preserved
    assert "115%" in window.resolution_label.text()

    window._on_wheel_zoom(-1, cx, cy)  # zoom out back to ~1.0
    assert window._preview_zoom_factor == pytest.approx(1.0)
    assert window._preview_source is source
    # Back at 100% the combo re-syncs to the preset (not blank).
    assert window.zoom_combo.currentText() == "100%"


def test_wheel_zoom_clamps_at_bounds(window):
    _load(window)
    cx = window.preview_image_label.width() / 2.0
    cy = window.preview_image_label.height() / 2.0

    for _ in range(200):
        window._on_wheel_zoom(1, cx, cy)
    assert window._preview_zoom_factor == pytest.approx(MAX_ZOOM)

    for _ in range(400):
        window._on_wheel_zoom(-1, cx, cy)
    assert window._preview_zoom_factor == pytest.approx(MIN_ZOOM)


def test_wheel_zoom_anchors_at_cursor(window):
    _load(window)
    window.preview_image_label.resize(300, 300)
    window._refresh_preview_view()
    window._on_wheel_zoom(1, 200.0, 150.0)  # 50px right of centre
    # zoom_ratio 1.15, cursor 50px right of centre -> offset shifts left.
    assert window._view_offset_x == pytest.approx(50.0 * (1.0 - ZOOM_STEP))
    assert window._view_offset_y == pytest.approx(0.0)


def test_wheel_zoom_noops_without_preview(window):
    assert window._preview_source is None
    window._on_wheel_zoom(1, 10.0, 10.0)
    assert window._preview_zoom_factor == 1.0
    assert window._view_offset_x == 0.0


# --------------------------------------------------------------------------
# (2) left-drag pan: viewport offset (unbounded, Tk clamps nothing)
# --------------------------------------------------------------------------
def test_drag_pan_changes_offset_unbounded(window):
    source = _load(window)
    window._on_pan_delta(30.0, -12.0)
    assert window._view_offset_x == 30.0
    assert window._view_offset_y == -12.0
    window._on_pan_delta(5.0, 5.0)
    assert window._view_offset_x == 35.0
    assert window._view_offset_y == -7.0
    # No clamping: offsets may grow arbitrarily (documented rule = Tk parity).
    window._on_pan_delta(10000.0, 10000.0)
    assert window._view_offset_x == 10035.0
    assert window._view_offset_y == 9993.0
    assert window._preview_source is source


def test_pan_composes_into_viewport_canvas(window):
    _load(window)
    window.preview_image_label.resize(300, 300)
    window._refresh_preview_view()
    assert window.preview_image_label.pixmap().width() == 64  # 100%, no pan
    assert window.preview_image_label.pixmap().height() == 32

    window._on_pan_delta(10.0, 10.0)
    assert window.preview_image_label.pixmap().width() == 300  # viewport canvas
    assert window.preview_image_label.pixmap().height() == 300


def test_pan_noops_without_preview(window):
    window._on_pan_delta(5.0, 5.0)
    assert window._view_offset_x == 0.0
    assert window._view_offset_y == 0.0


# --------------------------------------------------------------------------
# (3) zoom_combo interaction rule
# --------------------------------------------------------------------------
def test_combo_preset_sets_factor_and_recentres(window):
    _load(window)
    window._on_pan_delta(10.0, 10.0)
    assert window._view_offset_x == 10.0

    window.zoom_combo.setCurrentText("200%")
    assert window._preview_zoom_factor == 2.0
    assert window._view_offset_x == 0.0  # preset pick recentres
    assert window._view_offset_y == 0.0

    window.zoom_combo.setCurrentText("50%")
    assert window._preview_zoom_factor == 0.5

    window.zoom_combo.setCurrentText("100%")
    assert window._preview_zoom_factor == 1.0


def test_wheel_zoom_blank_combo_for_custom_factor(window):
    _load(window)
    cx = window.preview_image_label.width() / 2.0
    cy = window.preview_image_label.height() / 2.0

    window._on_wheel_zoom(1, cx, cy)  # 1.15 -> not a preset -> blank combo
    assert window.zoom_combo.currentText() == ""
    assert window._preview_zoom_factor == pytest.approx(1.15)

    # A subsequent combo pick returns to a preset and resets the factor.
    window.zoom_combo.setCurrentText("100%")
    assert window._preview_zoom_factor == 1.0
    assert window.zoom_combo.currentText() == "100%"


def test_wheel_zoom_exits_fit_and_continues_from_fit_scale(window):
    _load(window, width=128, height=64)
    window.preview_image_label.resize(256, 256)
    window.zoom_combo.setCurrentText("Fit")
    # Fit scale for 128x64 into 256x256 -> min(256/128, 256/64) = 2.0.
    assert window.zoom_combo.currentText() == "Fit"

    cx = window.preview_image_label.width() / 2.0
    cy = window.preview_image_label.height() / 2.0
    window._on_wheel_zoom(1, cx, cy)
    # Exits Fit: factor = fit_scale * 1.15 = 2.0 * 1.15.
    assert window._preview_zoom_factor == pytest.approx(2.0 * ZOOM_STEP)
    assert window.zoom_combo.currentText() == ""


# --------------------------------------------------------------------------
# (4) reset semantics: new preview image + Zoom Fit
# --------------------------------------------------------------------------
def test_pan_zoom_resets_on_new_preview_image(window):
    source_a = _load(window, name="a")
    cx = window.preview_image_label.width() / 2.0
    cy = window.preview_image_label.height() / 2.0
    window._on_wheel_zoom(1, cx, cy)
    window._on_pan_delta(20.0, 20.0)
    assert window._preview_zoom_factor != 1.0
    assert window._view_offset_x == 20.0

    # A new preview image resets zoom + pan to the defaults.
    source_b = _load(window, width=80, height=40, name="b")
    assert source_b is not source_a
    assert window._preview_zoom_factor == 1.0
    assert window._view_offset_x == 0.0
    assert window._view_offset_y == 0.0
    assert window.zoom_combo.currentText() == "100%"


def test_zoom_fit_resets_pan(window):
    _load(window)
    window._on_pan_delta(15.0, -15.0)
    assert window._view_offset_x == 15.0

    window.zoom_combo.setCurrentText("Fit")
    assert window._view_offset_x == 0.0
    assert window._view_offset_y == 0.0
    assert window.zoom_combo.currentText() == "Fit"


def test_rotate_resets_pan_but_keeps_zoom(window):
    _load(window)
    cx = window.preview_image_label.width() / 2.0
    cy = window.preview_image_label.height() / 2.0
    window._on_wheel_zoom(1, cx, cy)
    factor = window._preview_zoom_factor
    window._on_pan_delta(10.0, 10.0)
    assert window._view_offset_x == 10.0

    window.rotate_right_button.click()
    assert window._view_offset_x == 0.0
    assert window._view_offset_y == 0.0
    # Rotation preserves the continuous zoom factor (Tk parity).
    assert window._preview_zoom_factor == factor


# --------------------------------------------------------------------------
# (5) localization: the zoom label is numeric-only (no new FR/EN keys)
# --------------------------------------------------------------------------
def test_zoom_label_is_language_neutral(window):
    _load(window)
    cx = window.preview_image_label.width() / 2.0
    cy = window.preview_image_label.height() / 2.0
    window._on_wheel_zoom(1, cx, cy)
    assert "115%" in window.resolution_label.text()

    # Switching language must not change the numeric zoom label.
    window.language_combo.setCurrentText("Français")
    assert "115%" in window.resolution_label.text()
    window.language_combo.setCurrentText("English")
    assert "115%" in window.resolution_label.text()


def test_no_new_translation_keys_needed():
    """Pan/zoom adds no visible localized string, so no new keys are required."""
    from seestar.gui_qt import localization

    # The existing zoom labels (Fit/100%/200%/50%) are language-neutral and the
    # numeric zoom label is computed from the factor, not a translation key.
    assert localization.translate("zoom_label", "en") == "Zoom:"
    # No "pan" / "zoom_step" / "zoom_percent" keys were introduced.
    for key in ("pan_label", "zoom_step", "zoom_percent", "wheel_zoom"):
        assert key not in localization.TRANSLATIONS


# --------------------------------------------------------------------------
# (6) engine-coupled absence assertions
# --------------------------------------------------------------------------
def test_preview_image_view_module_is_tk_and_engine_free():
    from pathlib import Path

    import seestar.gui_qt as gui_qt

    pkg_dir = Path(gui_qt.__file__).resolve().parent
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
    for name in ("preview_image_view.py", "preview_view.py"):
        text = (pkg_dir / name).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{name} references {token}"


def test_pan_zoom_does_not_touch_engine_or_source(window):
    """Wheel-zoom / pan never change ``_preview_source`` or the backend kwargs."""
    source = _load(window)
    cx = window.preview_image_label.width() / 2.0
    cy = window.preview_image_label.height() / 2.0
    window._on_wheel_zoom(1, cx, cy)
    window._on_pan_delta(3.0, 3.0)

    assert window._preview_source is source
    kw = window.build_run_request().backend_kwargs
    assert "zoom_factor" not in kw
    assert "pan_offset" not in kw
    assert "preview_zoom" not in kw


def test_preview_image_view_is_a_signal_emitting_label(window):
    """The preview surface is a ``QLabel`` subclass emitting the two gestures."""
    from PySide6.QtWidgets import QLabel

    label = window.preview_image_label
    assert isinstance(label, PreviewImageView)
    assert isinstance(label, QLabel)
    assert hasattr(label, "wheelZoom")
    assert hasattr(label, "panDelta")


# --------------------------------------------------------------------------
# Offscreen smoke: real widget event delivery (wheel + drag)
# --------------------------------------------------------------------------
def _send_wheel(widget, delta_y: int) -> None:
    """Deliver a mouse-wheel event with the given vertical angle delta."""
    # Exact geometric centre (floating) so the wheel zoom is *not* cursor-
    # anchored (rel_x/rel_y == 0) — deterministic for the smoke assertions.
    pos = QPointF(widget.width() / 2.0, widget.height() / 2.0)
    event = QWheelEvent(
        pos,
        pos,
        QPoint(0, 0),
        QPoint(0, delta_y),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


def test_widget_wheel_event_zooms_and_preserves_source(window, qapp):
    source = _load(window, name="smoke")
    label = window.preview_image_label
    label.resize(200, 100)
    window._refresh_preview_view()
    label.show()
    qapp.processEvents()

    _send_wheel(label, 120)  # scroll up -> zoom in
    qapp.processEvents()
    assert window._preview_zoom_factor == pytest.approx(1.15)
    assert window._preview_source is source

    _send_wheel(label, -120)  # scroll down -> zoom out
    qapp.processEvents()
    assert window._preview_zoom_factor == pytest.approx(1.0)
    assert window._preview_source is source


def test_widget_drag_event_pans(window, qapp):
    source = _load(window, name="drag")
    label = window.preview_image_label
    label.resize(200, 100)
    window._refresh_preview_view()
    label.show()
    qapp.processEvents()

    start = label.rect().center()
    target = QPoint(start.x() + 30, start.y() + 20)
    QTest.mousePress(label, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(label, target)
    QTest.mouseRelease(label, Qt.MouseButton.LeftButton, pos=target)
    qapp.processEvents()

    assert window._view_offset_x == pytest.approx(30.0)
    assert window._view_offset_y == pytest.approx(20.0)
    assert window._preview_source is source


def test_offscreen_smoke_simulated_backend_wheel_and_drag(qapp):
    """Explicit smoke: ``backend_mode="simulated"``, wheel-zoom + drag events."""
    win = MainWindow(backend_mode="simulated")
    try:
        source = _load(win, name="smoke-sim")
        assert win.backend_mode == "simulated"
        label = win.preview_image_label
        label.resize(256, 256)
        win._refresh_preview_view()

        # Wheel zoom in -> factor 1.15, source identity preserved.
        _send_wheel(label, 120)
        qapp.processEvents()
        assert win._preview_zoom_factor == pytest.approx(1.15)
        assert win._preview_source is source

        # Left-drag pan -> viewport offset changes, source identity preserved.
        start = label.rect().center()
        target = QPoint(start.x() + 25, start.y() - 10)
        QTest.mousePress(label, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(label, target)
        QTest.mouseRelease(label, Qt.MouseButton.LeftButton, pos=target)
        qapp.processEvents()
        assert win._view_offset_x == pytest.approx(25.0)
        assert win._view_offset_y == pytest.approx(-10.0)
        assert win._preview_source is source
    finally:
        win.shutdown()
