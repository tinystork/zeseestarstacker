"""M16 seam tests: Stacking tab content parity + closure.

Offscreen tests for the Stacking-tab closure (Tk ``tab_stacking`` → Qt):

* every Stacking-tab control (including the newly added ``Use GPU`` checkbox
  and the ``HQ RAM limit (GB)`` spinbox) exists as a real widget with the Tk
  type / range / ``QtSettingsState`` default,
* the Enable-drizzle checkbox gates the drizzle mode / group-size / GPU and the
  Stacking-tab Drizzle-advanced sub-options (scale / WHT threshold / kernel /
  pixfrac) exactly like the Tk ``_update_drizzle_options_state`` method, and
  the group-size spinbox is enabled only in the Large-dataset (``Incremental``)
  mode,
* the "Apply Final SCNR" checkbox gates its target-channel / amount /
  preserve-luminosity sub-options (Tk ``_update_final_scnr_options_state``),
* the newly added labels localize FR/EN via the Qt-local ``localization``
  module,
* engine-coupled items (``use_gpu`` / ``max_hq_mem_gb``) exist in
  ``QtSettingsState``, round-trip through persistence, and (M20) are wired into
  the Qt run request as seam-only fields while ``build_backend_kwargs`` itself
  stays unchanged (Tk parity),
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
    QSpinBox,
)

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt import localization
from seestar.gui_qt.main_window import DRIZZLE_KERNELS
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


def _drizzle_advanced_widgets(window):
    """Return the four Drizzle-advanced widgets on the Stacking-tab Drizzle block."""
    return (
        window.drizzle_scale_spin,
        window.drizzle_wht_spin,
        window.drizzle_kernel_combo,
        window.drizzle_pixfrac_spin,
    )


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

    # Part C: the architecture hint ("Standard and Large dataset share the same
    # M3 accumulator...") was removed from the Qt UI; the attribute must not
    # exist anymore (the Tk oracle keeps its own hint, untouched).
    assert not hasattr(window, "drizzle_policy_hint")

    # Pre-existing Stacking-tab controls (Tk parity defaults / ranges).
    assert window.batch_spin.value() == 0
    assert window.stacking_mode_combo.currentText() == "kappa-sigma"
    assert window.final_combine_combo.currentText() == "Mean"
    assert window.drizzle_check.isChecked() is False
    assert window.drizzle_mode_combo.currentText() == "Standard"
    assert window.drizzle_group_spin.value() == 50
    assert window.drizzle_group_spin.minimum() == 1
    assert window.drizzle_group_spin.maximum() == 100_000
    assert window.drizzle_group_spin.singleStep() == 10
    assert window.solver_combo.currentText() == "none"


def test_stacking_drizzle_and_scnr_controls_have_tk_defaults(window):
    defaults = QtSettingsState.defaults()
    widgets = window._settings_widgets

    # Stacking-tab Drizzle-advanced sub-options (D3) share the Enable-drizzle
    # gate and keep the exact former Expert widget specs.
    assert isinstance(window.drizzle_scale_spin, QSpinBox)
    assert window.drizzle_scale_spin.value() == defaults["drizzle_scale"]  # 2
    assert window.drizzle_scale_spin.minimum() == 2
    assert window.drizzle_scale_spin.maximum() == 4
    assert window.drizzle_scale_spin.singleStep() == 1

    assert isinstance(window.drizzle_wht_spin, QDoubleSpinBox)
    assert window.drizzle_wht_spin.value() == pytest.approx(
        defaults["drizzle_wht_threshold"]
    )  # 0.0
    assert window.drizzle_wht_spin.minimum() == 0.0
    assert window.drizzle_wht_spin.maximum() == 1.0
    assert window.drizzle_wht_spin.decimals() == 3

    assert isinstance(window.drizzle_kernel_combo, QComboBox)
    assert window.drizzle_kernel_combo.currentText() == defaults["drizzle_kernel"]  # "square"
    assert [
        window.drizzle_kernel_combo.itemText(i)
        for i in range(window.drizzle_kernel_combo.count())
    ] == DRIZZLE_KERNELS

    assert isinstance(window.drizzle_pixfrac_spin, QDoubleSpinBox)
    assert window.drizzle_pixfrac_spin.value() == pytest.approx(
        defaults["drizzle_pixfrac"]
    )  # 1.0
    assert window.drizzle_pixfrac_spin.minimum() == 0.01
    assert window.drizzle_pixfrac_spin.maximum() == 2.0

    # No duplication: the four controls are no longer in the Expert surface.
    for attr in ("drizzle_scale", "drizzle_wht_threshold", "drizzle_kernel", "drizzle_pixfrac"):
        assert attr not in widgets, f"{attr} should not be in Expert _settings_widgets"

    # SCNR sub-options.
    assert widgets["apply_final_scnr"].isChecked() is defaults["apply_final_scnr"]  # True
    assert widgets["final_scnr_target_channel"].currentText() == "green"
    assert widgets["final_scnr_amount"].value() == pytest.approx(defaults["final_scnr_amount"])
    assert widgets["final_scnr_preserve_luminosity"].isChecked() is defaults["final_scnr_preserve_luminosity"]


# --------------------------------------------------------------------------
# R3b: drizzle mode labels (Standard / Large dataset) + scale x2/x3/x4
# --------------------------------------------------------------------------
def test_drizzle_mode_combo_shows_labels_not_backend_values(window):
    """The combo displays user-facing labels; itemData keeps the backend value."""
    combo = window.drizzle_mode_combo
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert labels == ["Standard", "Large dataset"]
    assert "Final" not in labels and "Incremental" not in labels
    assert [combo.itemData(i) for i in range(combo.count())] == ["Final", "Incremental"]
    assert combo.currentText() == "Standard"
    assert combo.currentData() == "Final"


def test_drizzle_mode_label_switches_value_to_state(window):
    """Selecting the user-facing "Large dataset" label yields "Incremental" state."""
    window.drizzle_check.setChecked(True)
    window.drizzle_mode_combo.setCurrentText("Large dataset")
    state = window.collect_settings_state()
    assert state.drizzle_mode == "Incremental"
    # Back to Standard yields the "Final" backend value.
    window.drizzle_mode_combo.setCurrentText("Standard")
    state = window.collect_settings_state()
    assert state.drizzle_mode == "Final"


def test_drizzle_scale_restricted_to_x2_x3_x4(window):
    """drizzle_scale cannot be x1 (widget range 2-4) and defaults to 2."""
    scale = window.drizzle_scale_spin
    assert isinstance(scale, QSpinBox)
    assert scale.minimum() == 2 and scale.maximum() == 4
    assert scale.value() == 2
    # x1 is not representable: setting 1 clamps to the minimum (2).
    scale.setValue(1)
    assert scale.value() == 2
    state = window.collect_settings_state()
    assert state.drizzle_scale == 2
    # x3 round-trips through the model and the backend kwargs as a float.
    scale.setValue(3)
    state = window.collect_settings_state()
    assert state.drizzle_scale == 3
    kw = window.build_run_request().backend_kwargs
    assert kw["drizzle_scale"] == 3.0


# --------------------------------------------------------------------------
# (1b) newly added labels localize FR/EN
# --------------------------------------------------------------------------
def test_stacking_new_labels_localize_fr_en(window):
    assert localization.translate("hq_ram_limit", "en") == "HQ RAM limit (GB)"
    assert localization.translate("hq_ram_limit", "fr") == "Limite RAM HQ (Go)"
    assert localization.translate("drizzle_use_gpu", "en") == "Use GPU"
    assert localization.translate("drizzle_use_gpu", "fr") == "Utiliser le GPU"

    # Part C: "Drizzle group size" label renamed to "Preview group size" on the
    # presentation side (state key ``drizzle_group_size`` unchanged).
    assert localization.translate("drizzle_group_size", "en") == "Preview group size"
    assert localization.translate("drizzle_group_size", "fr") == "Taille du groupe d'aperçu"

    # The GPU toggle label still re-labels on language switch.
    assert window.use_gpu_check.text() == "Use GPU"
    window.language_combo.setCurrentText("Français")
    assert window.use_gpu_check.text() == "Utiliser le GPU"
    window.language_combo.setCurrentText("English")
    assert window.use_gpu_check.text() == "Use GPU"


def test_stacking_localization_keys_have_full_parity():
    for key in ("hq_ram_limit", "drizzle_use_gpu", "drizzle_group_size"):
        entry = localization.TRANSLATIONS[key]
        assert set(entry) == {"en", "fr"}, key
        assert entry["en"] and entry["fr"], key


# --------------------------------------------------------------------------
# (2) drizzle enabler gating mirrors Tk + (3) group-size / mode interaction
# --------------------------------------------------------------------------
def test_drizzle_enabler_gates_suboptions(window):
    # Default: drizzle off -> every sub-option disabled.
    assert window.drizzle_check.isChecked() is False
    assert not window.drizzle_mode_combo.isEnabled()
    assert not window.drizzle_group_spin.isEnabled()
    assert not window.use_gpu_check.isEnabled()
    for w in _drizzle_advanced_widgets(window):
        assert not w.isEnabled()

    # Enable drizzle -> mode / GPU / advanced sub-options re-enable; the
    # group-size spin stays disabled in Standard (Final) mode.
    window.drizzle_check.setChecked(True)
    assert window.drizzle_mode_combo.isEnabled()
    assert window.use_gpu_check.isEnabled()
    for w in _drizzle_advanced_widgets(window):
        assert w.isEnabled()
    assert not window.drizzle_group_spin.isEnabled()

    # Large-dataset (Incremental) mode re-enables the group-size spin.
    window.drizzle_mode_combo.setCurrentText("Large dataset")
    assert window.drizzle_group_spin.isEnabled()

    # Back to Standard disables the group-size spin again.
    window.drizzle_mode_combo.setCurrentText("Standard")
    assert not window.drizzle_group_spin.isEnabled()

    # Disable drizzle -> everything gated off again.
    window.drizzle_check.setChecked(False)
    assert not window.drizzle_mode_combo.isEnabled()
    assert not window.use_gpu_check.isEnabled()
    assert not window.drizzle_group_spin.isEnabled()
    for w in _drizzle_advanced_widgets(window):
        assert not w.isEnabled()


def test_signed_wht_kernel_disables_threshold_without_losing_requested_value(window):
    """Lanczos marks WHT threshold N/A while preserving positive-kernel state."""
    window.drizzle_check.setChecked(True)
    window.drizzle_kernel_combo.setCurrentText("square")
    window.drizzle_wht_spin.setValue(0.7)
    assert window.drizzle_wht_spin.isEnabled()

    for kernel in ("lanczos2", "lanczos3"):
        window.drizzle_kernel_combo.setCurrentText(kernel)
        assert not window.drizzle_wht_spin.isEnabled()
        assert "signed" in window.drizzle_wht_spin.toolTip().lower()
        assert window.collect_settings_state().drizzle_wht_threshold == pytest.approx(0.7)

    window.language_combo.setCurrentText("Français")
    assert "signé" in window.drizzle_wht_spin.toolTip().lower()
    window.language_combo.setCurrentText("English")

    window.drizzle_kernel_combo.setCurrentText("square")
    assert window.drizzle_wht_spin.isEnabled()
    assert window.drizzle_wht_spin.toolTip() == ""
    assert window.drizzle_wht_spin.value() == pytest.approx(0.7)
    assert window.collect_settings_state().drizzle_wht_threshold == pytest.approx(0.7)


def test_group_size_mode_interaction_only_large_dataset(window):
    """Group size is enabled only for drizzle + Large-dataset (Incremental)."""
    window.drizzle_check.setChecked(True)
    window.drizzle_mode_combo.setCurrentText("Standard")
    assert not window.drizzle_group_spin.isEnabled()
    window.drizzle_mode_combo.setCurrentText("Large dataset")
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
# (5) engine-coupled items: collected in the model, wired into the run request
#     as seam-only fields (M20).  ``build_backend_kwargs`` itself is unchanged.
# --------------------------------------------------------------------------
def test_engine_coupled_items_are_wired_into_run_request(window):
    window.use_gpu_check.setChecked(True)
    window.max_hq_mem_spin.setValue(32)

    state = window.collect_settings_state()
    # They exist in the model (persisted/collected like Tk)...
    assert state.use_gpu is True
    assert state.max_hq_mem_gb == 32.0

    # ...and M20 wires them into the Qt run request as seam-only fields.  The
    # byte conversion (``max_hq_mem``) happens in the backend adapter, not in
    # the snapshot, so the request still carries the GB value.
    kw = window.build_run_request().backend_kwargs
    assert kw["use_gpu"] is True
    assert kw["max_hq_mem_gb"] == 32.0
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
    window.drizzle_mode_combo.setCurrentText("Large dataset")
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
    # Sanity: the two Stacking-tab checkboxes are present.  ``use_gpu_check``
    # moved to the System tab in M25.5-C (same state key + seam).
    assert len(stacking_children) >= 2  # boring_check + drizzle_check
