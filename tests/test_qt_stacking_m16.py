"""M16 seam tests: Stacking tab content parity + closure.

Offscreen tests for the Stacking-tab closure (Tk ``tab_stacking`` → Qt):

* every Stacking-tab control (including the newly added ``Use GPU`` checkbox,
  the ``HQ RAM limit (GB)`` spinbox and the drizzle-policy hint) exists as a
  real widget with the Tk type / range / ``QtSettingsState`` default,
* the Enable-drizzle checkbox gates the drizzle mode / group-size / GPU and the
  Expert-tab "Drizzle Advanced" sub-options exactly like the Tk
  ``_update_drizzle_options_state`` method, and the group-size spinbox is
  enabled only in the Large-dataset (``Incremental``) mode,
* the "Apply Final SCNR" checkbox gates its target-channel / amount /
  preserve-luminosity sub-options (Tk ``_update_final_scnr_options_state``),
* the newly added labels localize FR/EN via the Qt-local ``localization``
  module,
* engine-coupled items (``use_gpu`` / ``max_hq_mem_gb``) stay display-only:
  they exist in ``QtSettingsState`` and round-trip through persistence but are
  NOT consumed by ``build_backend_kwargs``,
* the Stacking tab has no reset button (the Tk Stacking tab has none).

No real stacking, no engine, no Tk.  ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QSpinBox,
)

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt import localization
from seestar.gui_qt.settings_state import QtSettingsState


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


def _drizzle_advanced_attrs():
    return ("drizzle_scale", "drizzle_wht_threshold", "drizzle_kernel", "drizzle_pixfrac")


# --------------------------------------------------------------------------
# (1) Stacking-tab controls exist with the right type / range / default
# --------------------------------------------------------------------------
def test_stacking_controls_exist_with_tk_ranges_and_defaults(window):
    defaults = QtSettingsState.defaults()

    # Newly added Stacking-tab controls.
    assert isinstance(window.use_gpu_check, QCheckBox)
    assert window.use_gpu_check.isChecked() is defaults["use_gpu"]  # False

    assert isinstance(window.max_hq_mem_spin, QSpinBox)
    assert window.max_hq_mem_spin.minimum() == 1
    assert window.max_hq_mem_spin.maximum() == 64
    assert window.max_hq_mem_spin.singleStep() == 1
    assert window.max_hq_mem_spin.value() == int(defaults["max_hq_mem_gb"])  # 8

    assert isinstance(window.drizzle_policy_hint, QLabel)
    assert window.drizzle_policy_hint.text() == localization.translate("drizzle_policy_hint", "en")

    # Pre-existing Stacking-tab controls (Tk parity defaults / ranges).
    assert window.batch_spin.value() == 0
    assert window.stacking_mode_combo.currentText() == "kappa-sigma"
    assert window.final_combine_combo.currentText() == "Mean"
    assert window.drizzle_check.isChecked() is False
    assert window.drizzle_mode_combo.currentText() == "Final"
    assert window.drizzle_group_spin.value() == 50
    assert window.drizzle_group_spin.minimum() == 1
    assert window.drizzle_group_spin.maximum() == 100_000
    assert window.drizzle_group_spin.singleStep() == 10
    assert window.solver_combo.currentText() == "none"


def test_expert_drizzle_and_scnr_controls_have_tk_defaults(window):
    defaults = QtSettingsState.defaults()
    widgets = window._settings_widgets

    # Expert-tab "Drizzle Advanced" sub-options share the Enable-drizzle gate.
    scale = widgets["drizzle_scale"]
    assert isinstance(scale, QSpinBox)
    assert scale.value() == defaults["drizzle_scale"]  # 2
    assert scale.minimum() == 1 and scale.maximum() == 10

    wht = widgets["drizzle_wht_threshold"]
    assert isinstance(wht, QDoubleSpinBox)
    assert wht.value() == pytest.approx(defaults["drizzle_wht_threshold"])  # 0.7

    kernel = widgets["drizzle_kernel"]
    assert isinstance(kernel, QComboBox)
    assert kernel.currentText() == defaults["drizzle_kernel"]  # "square"

    pixfrac = widgets["drizzle_pixfrac"]
    assert isinstance(pixfrac, QDoubleSpinBox)
    assert pixfrac.value() == pytest.approx(defaults["drizzle_pixfrac"])  # 1.0

    # SCNR sub-options.
    assert widgets["apply_final_scnr"].isChecked() is defaults["apply_final_scnr"]  # True
    assert widgets["final_scnr_target_channel"].currentText() == "green"
    assert widgets["final_scnr_amount"].value() == pytest.approx(defaults["final_scnr_amount"])
    assert widgets["final_scnr_preserve_luminosity"].isChecked() is defaults["final_scnr_preserve_luminosity"]


# --------------------------------------------------------------------------
# (1b) newly added labels localize FR/EN
# --------------------------------------------------------------------------
def test_stacking_new_labels_localize_fr_en(window):
    assert localization.translate("hq_ram_limit", "en") == "HQ RAM limit (GB)"
    assert localization.translate("hq_ram_limit", "fr") == "Limite RAM HQ (Go)"
    assert localization.translate("drizzle_use_gpu", "en") == "Use GPU"
    assert localization.translate("drizzle_use_gpu", "fr") == "Utiliser le GPU"

    en_hint = localization.translate("drizzle_policy_hint", "en")
    fr_hint = localization.translate("drizzle_policy_hint", "fr")
    assert en_hint and fr_hint and en_hint != fr_hint

    # The live hint label re-labels on language switch.
    assert window.drizzle_policy_hint.text() == en_hint
    assert window.use_gpu_check.text() == "Use GPU"
    window.language_combo.setCurrentText("Français")
    assert window.drizzle_policy_hint.text() == fr_hint
    assert window.use_gpu_check.text() == "Utiliser le GPU"
    window.language_combo.setCurrentText("English")
    assert window.drizzle_policy_hint.text() == en_hint
    assert window.use_gpu_check.text() == "Use GPU"


def test_stacking_localization_keys_have_full_parity():
    for key in ("hq_ram_limit", "drizzle_use_gpu", "drizzle_policy_hint"):
        entry = localization.TRANSLATIONS[key]
        assert set(entry) == {"en", "fr"}, key
        assert entry["en"] and entry["fr"], key


# --------------------------------------------------------------------------
# (2) drizzle enabler gating mirrors Tk + (3) group-size / mode interaction
# --------------------------------------------------------------------------
def test_drizzle_enabler_gates_suboptions(window):
    widgets = window._settings_widgets

    # Default: drizzle off -> every sub-option disabled.
    assert window.drizzle_check.isChecked() is False
    assert not window.drizzle_mode_combo.isEnabled()
    assert not window.drizzle_group_spin.isEnabled()
    assert not window.use_gpu_check.isEnabled()
    for attr in _drizzle_advanced_attrs():
        assert not widgets[attr].isEnabled(), attr

    # Enable drizzle -> mode / GPU / advanced sub-options re-enable; the
    # group-size spin stays disabled in Standard (Final) mode.
    window.drizzle_check.setChecked(True)
    assert window.drizzle_mode_combo.isEnabled()
    assert window.use_gpu_check.isEnabled()
    for attr in _drizzle_advanced_attrs():
        assert widgets[attr].isEnabled(), attr
    assert not window.drizzle_group_spin.isEnabled()

    # Large-dataset (Incremental) mode re-enables the group-size spin.
    window.drizzle_mode_combo.setCurrentText("Incremental")
    assert window.drizzle_group_spin.isEnabled()

    # Back to Standard disables the group-size spin again.
    window.drizzle_mode_combo.setCurrentText("Final")
    assert not window.drizzle_group_spin.isEnabled()

    # Disable drizzle -> everything gated off again.
    window.drizzle_check.setChecked(False)
    assert not window.drizzle_mode_combo.isEnabled()
    assert not window.use_gpu_check.isEnabled()
    assert not window.drizzle_group_spin.isEnabled()
    for attr in _drizzle_advanced_attrs():
        assert not widgets[attr].isEnabled(), attr


def test_group_size_mode_interaction_only_large_dataset(window):
    """Group size is enabled only for drizzle + Large-dataset (Incremental)."""
    window.drizzle_check.setChecked(True)
    window.drizzle_mode_combo.setCurrentText("Final")
    assert not window.drizzle_group_spin.isEnabled()
    window.drizzle_mode_combo.setCurrentText("Incremental")
    assert window.drizzle_group_spin.isEnabled()
    # Group size value still propagates regardless of enablement (backend key).
    window.drizzle_group_spin.setValue(77)
    state = window.collect_settings_state()
    assert state.drizzle_group_size == 77


# --------------------------------------------------------------------------
# SCNR enabler gating mirrors Tk
# --------------------------------------------------------------------------
def test_scnr_enabler_gating(window):
    widgets = window._settings_widgets
    gated = (
        "final_scnr_target_channel",
        "final_scnr_amount",
        "final_scnr_preserve_luminosity",
    )

    # Default: apply_final_scnr=True -> sub-options enabled.
    assert widgets["apply_final_scnr"].isChecked() is True
    for attr in gated:
        assert widgets[attr].isEnabled(), attr

    widgets["apply_final_scnr"].setChecked(False)
    for attr in gated:
        assert not widgets[attr].isEnabled(), attr

    widgets["apply_final_scnr"].setChecked(True)
    for attr in gated:
        assert widgets[attr].isEnabled(), attr


# --------------------------------------------------------------------------
# (5) engine-coupled items are display-only (not in build_backend_kwargs)
# --------------------------------------------------------------------------
def test_engine_coupled_items_are_display_only(window):
    window.use_gpu_check.setChecked(True)
    window.max_hq_mem_spin.setValue(32)

    state = window.collect_settings_state()
    # They exist in the model (persisted/collected like Tk)...
    assert state.use_gpu is True
    assert state.max_hq_mem_gb == 32.0

    # ...but they are NOT wired to the backend.
    kw = window.build_run_request().backend_kwargs
    assert "use_gpu" not in kw
    assert "max_hq_mem_gb" not in kw
    assert "max_hq_mem" not in kw


def test_engine_coupled_fields_persist_round_trip():
    defaults = QtSettingsState.defaults()
    assert defaults["use_gpu"] is False
    assert defaults["max_hq_mem_gb"] == 8.0

    state = QtSettingsState()
    state.use_gpu = True
    state.max_hq_mem_gb = 12.0
    restored = QtSettingsState.from_dict(state.to_dict())
    assert restored.use_gpu is True
    assert restored.max_hq_mem_gb == 12.0


def test_engine_coupled_fields_do_not_mutate_preview_source(window):
    """Exercising the new controls never touches ``_preview_source``."""
    before = window._preview_source
    window.use_gpu_check.setChecked(True)
    window.max_hq_mem_spin.setValue(16)
    window.drizzle_check.setChecked(True)
    window.drizzle_mode_combo.setCurrentText("Incremental")
    window.drizzle_group_spin.setValue(99)
    window.collect_settings_state()
    assert window._preview_source is before


# --------------------------------------------------------------------------
# (4) reset behaviour — the Tk Stacking tab has none
# --------------------------------------------------------------------------
def test_stacking_tab_has_no_reset_button(window):
    """Tk ``tab_stacking`` has no reset button (only the Expert tab does)."""
    # The only reset action lives on the Expert tab; the Stacking tab exposes
    # no equivalent.
    assert window.reset_expert_button is not None
    stacking_children = window._stacking_tab.findChildren(QCheckBox)
    # No "Reset" push button on the Stacking tab.
    from PySide6.QtWidgets import QPushButton

    stacking_buttons = window._stacking_tab.findChildren(QPushButton)
    reset_texts = [b.text().lower() for b in stacking_buttons]
    assert not any("reset" in t for t in reset_texts)
    # Sanity: the three Stacking-tab checkboxes are all present.
    assert len(stacking_children) >= 3  # boring_check + drizzle_check + use_gpu_check
