"""D4 Qt tests: the signed-Lanczos float32 hold on the real stacking tab.

Signed Lanczos kernels (``lanczos2`` / ``lanczos3``) legitimately produce
negative ringing, which a uint16 export would silently clip.  While Drizzle is
effectively enabled AND such a kernel is selected, ``_update_drizzle_gating``
therefore auto-checks the Expert-tab "Save final as float32" checkbox and HOLDS
it disabled with a localized EN/FR tooltip; switching back to a non-signed
kernel (square/gaussian/point/turbo) re-enables it and restores the previously
persisted value.

These tests pin, on a REAL offscreen ``MainWindow``:

* selecting ``lanczos2`` with Drizzle enabled forces the checkbox checked +
  disabled (localized tooltip); ``lanczos3`` behaves identically;
* every non-signed kernel re-enables the checkbox and restores the prior value
  (both prior False and prior True);
* the transmitted value always matches the checkbox (state +
  ``build_run_request`` backend kwarg), so the GUI never lies to the engine;
* toggling Drizzle off / boring-on releases the hold (Classic science: no
  Lanczos ringing to protect);
* the new localization key carries full en/fr parity.

No stacking, no engine, no FITS.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QCheckBox, QApplication

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt import localization

TOOLTIP_KEY = "".join(
    ("save", "_as", "_float32", "_signed", "_lanczos", "_tooltip")
)
EN_TOOLTIP = localization.TRANSLATIONS[TOOLTIP_KEY]["en"]
FR_TOOLTIP = localization.TRANSLATIONS[TOOLTIP_KEY]["fr"]

SIGNED_KERNELS = ("lanczos2", "lanczos3")
NON_SIGNED_KERNELS = ("square", "gaussian", "point", "turbo")


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


def _float32_widget(window):
    w = window._settings_widgets["save_final_as_float32"]
    assert isinstance(w, QCheckBox)
    return w


def _request_carries_float32(window, expected: bool):
    state = window.collect_settings_state()
    assert state.save_final_as_float32 is expected
    request = window.build_run_request()
    assert request.backend_kwargs["save_as_float32"] is expected
    assert request.backend_kwargs["use_drizzle"] is True


def test_localization_key_full_parity():
    entry = localization.TRANSLATIONS[TOOLTIP_KEY]
    assert set(entry) == {"en", "fr"}
    assert entry["en"] and entry["fr"]
    # The tooltip states the semantic truth: signed Lanczos requires float32.
    assert "float32" in entry["en"] and "Lanczos" in entry["en"]
    assert "float32" in entry["fr"] and "Lanczos" in entry["fr"]
    assert "clip" in entry["en"].lower()


@pytest.mark.parametrize("kernel", SIGNED_KERNELS)
def test_signed_lanczos_checks_and_holds_float32(window, kernel):
    widget = _float32_widget(window)
    # Default: drizzle off, kernel not signed -> checkbox free + unchecked.
    assert widget.isChecked() is False
    assert widget.isEnabled() is True
    assert widget.toolTip() == ""

    window.drizzle_check.setChecked(True)
    window.drizzle_kernel_combo.setCurrentText(kernel)
    assert window.drizzle_kernel_combo.currentText() == kernel

    # Held checked + disabled with the localized tooltip.
    assert widget.isChecked() is True
    assert widget.isEnabled() is False
    assert widget.toolTip() == EN_TOOLTIP
    # The request is truthful: the GUI transmits exactly what it shows.
    _request_carries_float32(window, True)


@pytest.mark.parametrize("kernel", NON_SIGNED_KERNELS)
def test_non_signed_kernel_releases_hold_and_restores_prior_false(window, kernel):
    widget = _float32_widget(window)
    window.drizzle_check.setChecked(True)

    # Enter the hold with a prior value of False (the default).
    window.drizzle_kernel_combo.setCurrentText("lanczos2")
    assert widget.isChecked() is True and widget.isEnabled() is False

    window.drizzle_kernel_combo.setCurrentText(kernel)
    assert widget.isEnabled() is True
    assert widget.isChecked() is False  # prior value restored
    assert widget.toolTip() == ""
    _request_carries_float32(window, False)


def test_switch_back_restores_prior_true(window):
    widget = _float32_widget(window)
    window.drizzle_check.setChecked(True)
    widget.setChecked(True)  # user preference: float32 always
    assert widget.isChecked() is True

    window.drizzle_kernel_combo.setCurrentText("lanczos3")
    assert widget.isChecked() is True  # already True -> stays True
    assert widget.isEnabled() is False

    window.drizzle_kernel_combo.setCurrentText("square")
    assert widget.isEnabled() is True
    assert widget.isChecked() is True  # prior True preserved
    _request_carries_float32(window, True)


def test_lanczos2_lanczos3_transition_keeps_hold_and_value_transmitted(window):
    widget = _float32_widget(window)
    window.drizzle_check.setChecked(True)
    window.drizzle_kernel_combo.setCurrentText("lanczos2")
    assert widget.isChecked() is True and widget.isEnabled() is False

    # Switching between the two signed kernels keeps the hold in place.
    window.drizzle_kernel_combo.setCurrentText("lanczos3")
    assert widget.isChecked() is True
    assert widget.isEnabled() is False
    assert widget.toolTip() == EN_TOOLTIP
    _request_carries_float32(window, True)

    # Back to a positive kernel: released again (restore the pre-hold value).
    window.drizzle_kernel_combo.setCurrentText("turbo")
    assert widget.isEnabled() is True
    assert widget.isChecked() is False
    assert widget.toolTip() == ""


def test_hold_tooltip_localizes_fr_en(window):
    widget = _float32_widget(window)
    window.drizzle_check.setChecked(True)
    window.drizzle_kernel_combo.setCurrentText("lanczos3")
    assert widget.toolTip() == EN_TOOLTIP

    window.language_combo.setCurrentText("Français")
    assert widget.toolTip() == FR_TOOLTIP
    # The hold survives the language refresh.
    assert widget.isChecked() is True and widget.isEnabled() is False

    window.language_combo.setCurrentText("English")
    assert widget.toolTip() == EN_TOOLTIP
    assert widget.isChecked() is True and widget.isEnabled() is False


def test_drizzle_off_or_boring_releases_hold(window):
    widget = _float32_widget(window)
    window.drizzle_check.setChecked(True)
    window.drizzle_kernel_combo.setCurrentText("lanczos2")
    assert widget.isEnabled() is False and widget.isChecked() is True

    # Drizzle off -> Classic science: no Lanczos ringing to protect.
    window.drizzle_check.setChecked(False)
    assert widget.isEnabled() is True
    assert widget.isChecked() is False
    assert widget.toolTip() == ""

    # Re-enter the hold, then boring mode forces Drizzle off -> released.
    window.drizzle_check.setChecked(True)
    assert widget.isEnabled() is False
    window.boring_check.setChecked(True)
    assert window.drizzle_check.isChecked() is False
    assert widget.isEnabled() is True
    assert widget.isChecked() is False
    assert widget.toolTip() == ""


def test_d15_stacking_gate_still_intact_with_d4_hold(window):
    """D4 must not disturb the D1.5 rejection-selector gating."""
    window.drizzle_check.setChecked(True)
    window.drizzle_kernel_combo.setCurrentText("lanczos3")
    assert window.stacking_mode_combo.isEnabled() is False
    assert window.stacking_mode_combo.toolTip() != ""

    window.drizzle_check.setChecked(False)
    assert window.stacking_mode_combo.isEnabled() is True
    assert window.stacking_mode_combo.toolTip() == ""
