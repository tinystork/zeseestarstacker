"""D1.5 tests: the rejection selector is never presented as meaningful under
Drizzle (real Qt stacking-tab gating).

Product-semantic gap closed: ``_update_drizzle_gating`` previously gated the
Drizzle sub-options but NOT ``stacking_mode_combo``, so the user could pick
winsorized-sigma-clip / kappa-sigma / median while Drizzle was enabled even
though the Drizzle path bypasses those reducers entirely (direct accumulation,
no rejection).

These tests pin, on a REAL offscreen ``MainWindow``:

* Drizzle checked  -> ``stacking_mode_combo`` DISABLED with the localized N/A
  tooltip (EN and, after a language switch, FR);
* Drizzle unchecked -> the combo is re-enabled and the tooltip cleared;
* only the *enabled state + tooltip* change: the combo keeps its currentText
  and ``collect_settings_state()`` / ``build_run_request()`` still transmit the
  exact same stacking-mode value (a Classic run keeps its requested reducer,
  and the Drizzle provenance records the requested mode faithfully);
* the boring single-batch route temporarily GATES the Enable-drizzle checkbox
  off (Phase B2: the user request is remembered and restored when boring mode
  is left), which re-enables the combo during the boring episode (a Classic
  reducer still applies there);
* the new localization key carries full en/fr parity.

No stacking, no engine, no FITS.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt import localization

TOOLTIP_KEY = "".join(("stacking", "_mode", "_drizzle", "_na", "_tooltip"))
EN_TOOLTIP = localization.TRANSLATIONS[TOOLTIP_KEY]["en"]
FR_TOOLTIP = localization.TRANSLATIONS[TOOLTIP_KEY]["fr"]


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


def test_localization_key_full_parity():
    entry = localization.TRANSLATIONS[TOOLTIP_KEY]
    assert set(entry) == {"en", "fr"}
    assert entry["en"] and entry["fr"]
    # The tooltip states the semantic truth: not applied + direct accumulation.
    assert "Drizzle" in entry["en"] and "direct accumulation" in entry["en"]
    assert "Drizzle" in entry["fr"] and "accumulation directe" in entry["fr"]


def test_drizzle_checked_disables_stacking_mode_with_na_tooltip(window):
    # Default: Drizzle off -> the stacking selector behaves as before.
    assert window.drizzle_check.isChecked() is False
    assert window.stacking_mode_combo.isEnabled() is True
    assert window.stacking_mode_combo.toolTip() == ""

    # Drizzle on -> the rejection selector is disabled and explains why.
    window.drizzle_check.setChecked(True)
    assert window.stacking_mode_combo.isEnabled() is False
    assert window.stacking_mode_combo.toolTip() == EN_TOOLTIP
    assert "not applied" in EN_TOOLTIP.lower()
    assert "direct accumulation" in EN_TOOLTIP

    # The selector's VALUE is untouched: only the enabled state changed.
    assert window.stacking_mode_combo.currentText() == "kappa-sigma"

    # Drizzle off -> re-enabled and tooltip cleared (prior behaviour restored).
    window.drizzle_check.setChecked(False)
    assert window.stacking_mode_combo.isEnabled() is True
    assert window.stacking_mode_combo.toolTip() == ""


def test_stacking_mode_tooltip_localizes_fr_en(window):
    window.drizzle_check.setChecked(True)
    assert window.stacking_mode_combo.isEnabled() is False

    window.language_combo.setCurrentText("Français")
    assert window.stacking_mode_combo.toolTip() == FR_TOOLTIP
    assert "non appliqué" in FR_TOOLTIP.lower()
    assert "accumulation directe" in FR_TOOLTIP

    window.language_combo.setCurrentText("English")
    assert window.stacking_mode_combo.toolTip() == EN_TOOLTIP
    # A fresh language refresh keeps the gating + tooltip consistent.
    assert window.stacking_mode_combo.isEnabled() is False


def test_any_stacking_mode_value_is_still_transmitted_unchanged(window):
    """Gating must not change what is transmitted (state + request)."""
    # Try every Classic reducer offered by the combo while Drizzle is on.
    for mode in ("mean", "median", "kappa-sigma", "winsorized-sigma-clip"):
        window.stacking_mode_combo.setCurrentText(mode)
        window.drizzle_check.setChecked(True)
        assert window.stacking_mode_combo.isEnabled() is False

        state = window.collect_settings_state()
        assert state.stacking_mode == mode  # value survives the disabled state

        request = window.build_run_request()
        assert request.backend_kwargs["stacking_mode"] == mode
        assert request.backend_kwargs["use_drizzle"] is True

    # After disabling Drizzle the transmitted value is still the user choice.
    window.drizzle_check.setChecked(False)
    assert window.stacking_mode_combo.isEnabled() is True
    assert window.collect_settings_state().stacking_mode == "winsorized-sigma-clip"


def test_boring_route_gates_drizzle_and_reenables_selector(window):
    # Boring (single-batch) mode gates the Enable-drizzle checkbox (visually
    # cleared while the episode is active, request remembered), so the Classic
    # reducer applies and the selector must be usable again during the boring
    # episode.
    window.drizzle_check.setChecked(True)
    assert window.stacking_mode_combo.isEnabled() is False
    assert window.stacking_mode_combo.toolTip() != ""

    window.boring_check.setChecked(True)
    assert window.drizzle_check.isChecked() is False
    assert window.stacking_mode_combo.isEnabled() is True
    assert window.stacking_mode_combo.toolTip() == ""

    # Leaving boring restores the requested drizzle NON-destructively, so the
    # selector is gated again — no manual re-check is needed and the user
    # request survived the whole boring episode.
    window.boring_check.setChecked(False)
    assert window.drizzle_check.isChecked() is True
    assert window.stacking_mode_combo.isEnabled() is False
    assert window.stacking_mode_combo.toolTip() != ""


def test_drizzle_suboption_gating_still_intact_with_stacking_gate(window):
    """D1.5 must not disturb the existing drizzle sub-option gating (M16)."""
    assert not window.drizzle_mode_combo.isEnabled()
    window.drizzle_check.setChecked(True)
    assert window.drizzle_mode_combo.isEnabled()
    window.drizzle_check.setChecked(False)
    assert not window.drizzle_mode_combo.isEnabled()
    assert window.stacking_mode_combo.isEnabled() is True
