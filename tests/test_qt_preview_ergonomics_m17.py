"""M17 seam tests: right-panel preview ergonomics parity closure.

Offscreen tests for the M17 lot:

* the Res 1/1..1/4 preview-resolution cycle button exists, cycles 1/2/3/4 with
  the Tk ``Res 1/N`` label, wraps back to 1/1, persists its factor in GUI
  state, and never touches ``_preview_source`` (identity before/after),
* the local display-only downsample seam (``preview_view`` ``render_view`` /
  ``downsampled_image``) scales the rendered preview down without mutating the
  source,
* the kappa/winsor show/hide logic mirrors the Tk ``_toggle_kappa_visibility``
  for every stacking method / final-combine combination (purely cosmetic: the
  backend kwargs keep carrying the values regardless of visibility),
* the new Res button localizes FR/EN via the Qt-local ``localization`` module,
* the engine-coupled preview-downsample factor is absent from
  ``build_backend_kwargs`` (no engine/``queued_stacker`` call).

No real stacking, no engine, no Tk.  ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QFormLayout, QPushButton

from seestar.gui_qt import BackendPreviewPayload, MainWindow, create_application
from seestar.gui_qt import localization
from seestar.gui_qt.main_window import (
    DEFAULT_PREVIEW_RES_FACTOR,
    PREVIEW_RES_FACTORS,
)
from seestar.gui_qt.preview_view import downsampled_image, render_view


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


def _rgb(width: int, height: int, r: int, g: int, b: int) -> np.ndarray:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = r
    arr[:, :, 1] = g
    arr[:, :, 2] = b
    return arr


def _row_visible(win, attr: str) -> bool:
    """Whether a settings row is visible (QFormLayout row visibility)."""
    widget = win._settings_widgets[attr]
    form = win._settings_forms[attr]
    assert isinstance(form, QFormLayout)
    return form.isRowVisible(widget)


# --------------------------------------------------------------------------
# (1) Res 1/1..1/4 cycle button
# --------------------------------------------------------------------------
def test_res_button_exists_with_default_label_and_factor(window):
    assert isinstance(window.preview_res_button, QPushButton)
    assert window.preview_res_button.text() == "Res 1/1"
    assert window._preview_res_factor == DEFAULT_PREVIEW_RES_FACTOR == 1
    assert PREVIEW_RES_FACTORS == (1, 2, 3, 4)


def test_res_button_cycles_and_wraps(window):
    expected = ["Res 1/2", "Res 1/3", "Res 1/4", "Res 1/1"]
    factors = [2, 3, 4, 1]
    for label, factor in zip(expected, factors):
        window.preview_res_button.click()
        assert window.preview_res_button.text() == label
        assert window._preview_res_factor == factor
    # GUI state persisted across cycles (the factor is stored, not transient).
    assert window._preview_res_factor == 1


def test_res_button_does_not_touch_preview_source(window):
    """Cycling the Res button never mutates ``_preview_source`` (no preview)."""
    assert window._preview_source is None
    for _ in range(4):
        window.preview_res_button.click()
    assert window._preview_source is None


def test_res_button_preserves_preview_source_identity(window):
    """Cycling with a live preview keeps the stored source image identity."""
    window._on_preview(BackendPreviewPayload(data=_rgb(32, 16, 90, 90, 90), stack_name="res"))
    source = window._preview_source
    assert source is not None and not source.isNull()

    for _ in range(4):
        window.preview_res_button.click()

    assert window._preview_source is source
    assert window._preview_source is not None


# --------------------------------------------------------------------------
# (1b) local display-only downsample seam
# --------------------------------------------------------------------------
def test_downsampled_image_never_mutates_source():
    img = QImage(64, 64, QImage.Format.Format_RGB32)
    img.fill(0xFF102030)
    out = downsampled_image(img, 2)
    assert out is not img  # a fresh image, never the source
    assert (out.width(), out.height()) == (32, 32)
    # factor <= 1 returns the source unchanged.
    assert downsampled_image(img, 1) is img
    assert downsampled_image(img, 0) is img
    assert downsampled_image(img, -3) is img


def test_render_view_applies_downsample_factor():
    img = QImage(64, 64, QImage.Format.Format_RGB32)
    img.fill(0xFF000000)
    full = render_view(img, 0, "100%", None, downsample_factor=1)
    half = render_view(img, 0, "100%", None, downsample_factor=2)
    quarter = render_view(img, 0, "100%", None, downsample_factor=4)
    assert full.width() == 64
    assert half.width() == 32
    assert quarter.width() == 16
    # The source image is untouched by any of the renders.
    assert img.width() == 64 and img.height() == 64


def test_window_preview_actually_rerenders_at_new_factor(window):
    window._on_preview(BackendPreviewPayload(data=_rgb(64, 32, 0, 128, 255), stack_name="res2"))
    window.zoom_combo.setCurrentText("100%")
    # Default factor is 1 -> native size (no downsample).
    assert window.preview_image_label.pixmap().width() == 64

    # Cycle to factor 4 -> displayed width is 1/4 of the source.
    window.preview_res_button.click()  # -> 2
    assert window.preview_image_label.pixmap().width() == 32
    window.preview_res_button.click()  # -> 3
    window.preview_res_button.click()  # -> 4
    assert window._preview_res_factor == 4
    assert window.preview_image_label.pixmap().width() == 16


# --------------------------------------------------------------------------
# (2) kappa / winsor visibility mirrors the Tk rule
# --------------------------------------------------------------------------
def test_kappa_winsor_visibility_mirrors_tk(window):
    """Exercise every stacking method + the winsorized final-combine override."""
    # (stacking method, final-combine label) -> (show_kappa, show_winsor)
    cases = [
        ("kappa-sigma", "Mean", True, False),
        ("winsorized-sigma-clip", "Mean", True, True),
        ("mean", "Mean", False, False),
        ("median", "Mean", False, False),
        ("linear-fit-clip", "Mean", False, False),
        ("classic", "Mean", False, False),
        # Final-combine winsorized overrides: kappa AND winsor both shown even
        # when the stacking method is plain (Tk ``_toggle_kappa_visibility``).
        ("mean", "Winsorized Sigma Clip", True, True),
        ("kappa-sigma", "Winsorized Sigma Clip", True, True),
    ]
    for method, final_label, show_kappa, show_winsor in cases:
        window.stacking_mode_combo.setCurrentText(method)
        window.final_combine_combo.setCurrentText(final_label)
        assert _row_visible(window, "stack_kappa_low") is show_kappa, (method, final_label)
        assert _row_visible(window, "stack_kappa_high") is show_kappa, (method, final_label)
        assert _row_visible(window, "stack_winsor_limits") is show_winsor, (method, final_label)
        # The standalone "Kappa" field is not part of the Tk kappa frame.
        assert _row_visible(window, "kappa") is True, (method, final_label)


def test_kappa_winsor_visibility_is_cosmetic_only(window):
    """Hiding the widgets never removes their backend kwargs / values."""
    window.stacking_mode_combo.setCurrentText("mean")  # hides kappa + winsor
    window.final_combine_combo.setCurrentText("Mean")
    assert _row_visible(window, "stack_kappa_low") is False
    assert _row_visible(window, "stack_winsor_limits") is False

    state = window.collect_settings_state()
    assert state.stack_kappa_low == 3.0
    assert state.stack_kappa_high == 3.0
    assert state.stack_winsor_limits == "0.05,0.05"

    kw = window.build_run_request().backend_kwargs
    assert kw["stack_kappa_low"] == 3.0
    assert kw["stack_kappa_high"] == 3.0
    assert kw["winsor_limits"] == (0.05, 0.05)


# --------------------------------------------------------------------------
# (3) new right-panel control localizes FR/EN
# --------------------------------------------------------------------------
def test_preview_res_prefix_localizes(window):
    assert localization.translate("preview_res_prefix", "en") == "Res"
    assert localization.translate("preview_res_prefix", "fr") == "Rés"

    # Factor 1 (default) English label -> "1/1" form.
    assert window.preview_res_button.text() == "Res 1/1"

    window.language_combo.setCurrentText("Français")
    assert window.preview_res_button.text() == "Rés 1/1"

    # Factor 2 -> "1/2" form (French).
    window._preview_res_factor = 2
    window._render_preview_res_button()
    assert window.preview_res_button.text() == "Rés 1/2"

    window.language_combo.setCurrentText("English")
    assert window.preview_res_button.text() == "Res 1/2"


def test_preview_res_localization_key_has_full_parity():
    entry = localization.TRANSLATIONS["preview_res_prefix"]
    assert set(entry) == {"en", "fr"}
    assert entry["en"] and entry["fr"]


# --------------------------------------------------------------------------
# (4) engine-coupled items stay out of build_backend_kwargs
# --------------------------------------------------------------------------
def test_preview_res_factor_not_in_backend_kwargs(window):
    """The display-only Res factor is never sent to the backend."""
    window.preview_res_button.click()
    kw = window.build_run_request().backend_kwargs
    assert "preview_downsample_factor" not in kw
    assert "preview_res_factor" not in kw
    # It is not even a settings-state field (pure GUI state).
    assert not hasattr(window.collect_settings_state(), "preview_res_factor")
