"""M15 seam tests: Expert tab content parity + M14 histogram leftovers.

Offscreen tests for the Expert-tab closure and the M14 leftovers:

* every Expert-tab control (Tk ``tab_expert`` parity) exists as a real widget
  with the correct type, Tk range and ``QtSettingsState`` default,
* the BN / CB / final-crop / Photutils / coverage / low-weight enabler
  checkboxes gate their sub-option widgets exactly like the Tk
  ``_update_*_options_state`` / ``_update_master_tile_crop_state`` methods,
* the "Reset Expert Settings" button restores every Expert-tab setting to its
  model default (GUI state only, never the engine/settings file/preview),
* Expert-tab field labels + the reset button + the warning banner localize
  FR/EN via the Qt-local ``localization`` module,
* the persistent histogram preserves a manual X zoom across ``set_data``
  refreshes (Tk ``freeze_x_range`` semantics) and resets on reset-view /
  reset-zoom / auto-zoom,
* the dead ``render_histogram_pixmap`` helper (and its now-unused colour
  palette) is gone from ``preview_adjust``.

No real stacking, no engine, no Tk.  ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QSpinBox,
)

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt import localization
from seestar.gui_qt.histogram_view import HistogramView
from seestar.gui_qt.preview_render import render_preview_image
from seestar.gui_qt.settings_state import QtSettingsState
from seestar.settings_migration import (
    CURRENT_SETTINGS_SCHEMA_VERSION,
    SETTINGS_SCHEMA_VERSION_KEY,
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


def _grad(width: int, height: int, low: int, high: int) -> np.ndarray:
    """A grayscale horizontal gradient (uint8 2D)."""
    return np.tile(np.linspace(low, high, width).astype(np.uint8), (height, 1))


def _value(widget):
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QSpinBox):
        return widget.value()
    if isinstance(widget, QDoubleSpinBox):
        return widget.value()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    raise TypeError(f"unsupported widget {type(widget)!r}")


def _set_value(widget, value):
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QSpinBox):
        widget.setValue(int(value))
    elif isinstance(widget, QDoubleSpinBox):
        widget.setValue(float(value))
    elif isinstance(widget, QComboBox):
        widget.setCurrentText(str(value))
    else:
        raise TypeError(f"unsupported widget {type(widget)!r}")


def _find_label(win, english_text):
    for lbl in win._settings_tab.findChildren(QLabel):
        if lbl.text() == english_text:
            return lbl
    return None


# --- Tk Expert-tab field spec (attr -> (widget type, Tk range, default)) ------
# Defaults are taken from ``QtSettingsState`` (aligned with the Tk
# ``SettingsManager.get_default_values``); ranges mirror the Tk spinboxes.
INT_FIELDS = {
    "bn_perc_low": (QSpinBox, (0, 40)),
    "bn_perc_high": (QSpinBox, (10, 95)),
    "cb_border_size": (QSpinBox, (5, 150)),
    "cb_blur_radius": (QSpinBox, (0, 50)),
    "photutils_bn_box_size": (QSpinBox, (16, 1024)),
    "photutils_bn_filter_size": (QSpinBox, (1, 15)),
    "low_wht_percentile": (QSpinBox, (1, 100)),
    "low_wht_soften_px": (QSpinBox, (32, 512)),
}
FLOAT_FIELDS = {
    "bn_std_factor": (QDoubleSpinBox, (0.5, 5.0)),
    "bn_min_gain": (QDoubleSpinBox, (0.1, 2.0)),
    "bn_max_gain": (QDoubleSpinBox, (1.0, 10.0)),
    "cb_min_b_factor": (QDoubleSpinBox, (0.1, 1.0)),
    "cb_max_b_factor": (QDoubleSpinBox, (1.0, 3.0)),
    "master_tile_crop_percent": (QDoubleSpinBox, (0.0, 25.0)),
    "final_edge_crop_percent": (QDoubleSpinBox, (0.0, 25.0)),
    "photutils_bn_sigma_clip": (QDoubleSpinBox, (1.0, 5.0)),
    "photutils_bn_exclude_percentile": (QDoubleSpinBox, (0.0, 100.0)),
}
BOOL_FIELDS = {
    "apply_bn": QCheckBox,
    "apply_cb": QCheckBox,
    "apply_final_crop": QCheckBox,
    "apply_master_tile_crop": QCheckBox,
    "apply_photutils_bn": QCheckBox,
    "apply_batch_feathering": QCheckBox,
    "apply_coverage_render": QCheckBox,
    "apply_low_wht_mask": QCheckBox,
}
COMBO_FIELDS = {
    "bn_grid_size_str": (["8x8", "16x16", "24x24", "32x32", "64x64"], "24x24"),
}


# --------------------------------------------------------------------------
# (1) Expert-tab controls exist with the right type / range / default
# --------------------------------------------------------------------------
def test_expert_controls_exist_with_tk_ranges_and_defaults(window):
    defaults = QtSettingsState.defaults()
    widgets = window._settings_widgets

    for attr, (wtype, (lo, hi)) in INT_FIELDS.items():
        assert attr in widgets, f"missing Expert int field {attr}"
        w = widgets[attr]
        assert isinstance(w, wtype), attr
        assert w.minimum() == lo and w.maximum() == hi, (attr, w.minimum(), w.maximum())
        assert w.value() == defaults[attr], (attr, w.value(), defaults[attr])

    for attr, (wtype, (lo, hi)) in FLOAT_FIELDS.items():
        assert attr in widgets, f"missing Expert float field {attr}"
        w = widgets[attr]
        assert isinstance(w, wtype), attr
        assert w.minimum() == pytest.approx(lo) and w.maximum() == pytest.approx(hi), (
            attr,
            w.minimum(),
            w.maximum(),
        )
        assert w.value() == pytest.approx(defaults[attr]), (attr, w.value(), defaults[attr])

    for attr, wtype in BOOL_FIELDS.items():
        assert attr in widgets, f"missing Expert bool field {attr}"
        w = widgets[attr]
        assert isinstance(w, wtype), attr
        assert w.isChecked() == defaults[attr], (attr, w.isChecked(), defaults[attr])

    for attr, (choices, default) in COMBO_FIELDS.items():
        assert attr in widgets, f"missing Expert combo field {attr}"
        w = widgets[attr]
        assert isinstance(w, QComboBox), attr
        assert [w.itemText(i) for i in range(w.count())] == choices, attr
        assert w.currentText() == default, attr


def test_expert_chrome_exists(window):
    assert window.expert_warning_label is not None
    assert window.reset_expert_button is not None
    assert window.expert_warning_label.text() == "Expert Settings!"
    assert window.reset_expert_button.text() == "Reset Expert Settings"


def test_cov06b_modern_coverage_surface_and_backend_wiring(window):
    widgets = window._settings_widgets
    titles = [g.title() for g in window._settings_tab.findChildren(QGroupBox)]

    assert "Coverage / Edge Reconstruction" in titles
    assert "apply_feathering" not in widgets
    assert "feather_blur_px" not in widgets
    assert isinstance(widgets["apply_batch_feathering"], QCheckBox)
    assert widgets["apply_batch_feathering"].isChecked() is True
    assert isinstance(widgets["apply_coverage_render"], QCheckBox)
    assert widgets["apply_coverage_render"].isChecked() is False
    assert _find_label(window, "Coverage support taper") is not None
    assert _find_label(window, "Coverage-aware final reconstruction") is not None

    widgets["apply_coverage_render"].setChecked(True)
    state = window.collect_settings_state()
    request = window.build_run_request()
    assert state.apply_coverage_render is True
    assert request.backend_kwargs["apply_coverage_render"] is True


def test_cov06b_qt_old_config_migrates_once_and_round_trips(qapp, tmp_path):
    path = tmp_path / "seestar_settings.json"
    path.write_text(
        json.dumps({"apply_feathering": True, "batch_size": 7}),
        encoding="utf-8",
    )

    first = MainWindow(settings_path=str(path))
    try:
        assert first.settings_state.apply_feathering is False
        migrated = json.loads(path.read_text(encoding="utf-8"))
        assert migrated["apply_feathering"] is False
        assert migrated[SETTINGS_SCHEMA_VERSION_KEY] == CURRENT_SETTINGS_SCHEMA_VERSION

        first._settings_widgets["apply_coverage_render"].setChecked(True)
        first._save_persisted_settings()
    finally:
        first.shutdown()

    saved_once = json.loads(path.read_text(encoding="utf-8"))
    second = MainWindow(settings_path=str(path))
    try:
        assert second.settings_state.apply_feathering is False
        assert second.settings_state.apply_coverage_render is True
        assert json.loads(path.read_text(encoding="utf-8")) == saved_once
    finally:
        second.shutdown()


# --------------------------------------------------------------------------
# (3) enabler gating mirrors Tk
# --------------------------------------------------------------------------
def test_enabler_gating_mirrors_tk(window):
    widgets = window._settings_widgets

    # Default: apply_photutils_bn=False -> its sub-options disabled.
    assert widgets["apply_photutils_bn"].isChecked() is False
    for attr in ("photutils_bn_box_size", "photutils_bn_filter_size",
                 "photutils_bn_sigma_clip", "photutils_bn_exclude_percentile"):
        assert not widgets[attr].isEnabled(), attr

    # Check it -> sub-options re-enabled.
    widgets["apply_photutils_bn"].setChecked(True)
    for attr in ("photutils_bn_box_size", "photutils_bn_filter_size",
                 "photutils_bn_sigma_clip", "photutils_bn_exclude_percentile"):
        assert widgets[attr].isEnabled(), attr

    # Default: apply_bn=True -> BN sub-options enabled; uncheck disables.
    assert widgets["apply_bn"].isChecked() is True
    assert widgets["bn_grid_size_str"].isEnabled()
    widgets["apply_bn"].setChecked(False)
    for attr in ("bn_grid_size_str", "bn_perc_low", "bn_perc_high",
                 "bn_std_factor", "bn_min_gain", "bn_max_gain"):
        assert not widgets[attr].isEnabled(), attr

    # Low-weight mask gating.
    assert widgets["apply_low_wht_mask"].isChecked() is False
    assert not widgets["low_wht_percentile"].isEnabled()
    assert not widgets["low_wht_soften_px"].isEnabled()
    widgets["apply_low_wht_mask"].setChecked(True)
    assert widgets["low_wht_percentile"].isEnabled()
    assert widgets["low_wht_soften_px"].isEnabled()


# --------------------------------------------------------------------------
# (2) reset-to-defaults restores every Expert-tab value
# --------------------------------------------------------------------------
def test_reset_expert_settings_restores_defaults(window):
    defaults = QtSettingsState.defaults()
    widgets = window._settings_widgets

    # Mutate every Expert-tab setting away from its default.
    for attr, (wtype, (lo, hi)) in {**INT_FIELDS, **FLOAT_FIELDS}.items():
        _set_value(widgets[attr], lo)  # set to the minimum (always != default)
    for attr in BOOL_FIELDS:
        widgets[attr].setChecked(not defaults[attr])
    widgets["bn_grid_size_str"].setCurrentText("8x8")

    # Sanity: at least one value is no longer the default before the reset.
    assert widgets["bn_perc_low"].value() != defaults["bn_perc_low"]

    window.reset_expert_button.click()

    for attr in INT_FIELDS:
        assert _value(widgets[attr]) == defaults[attr], attr
    for attr in FLOAT_FIELDS:
        assert _value(widgets[attr]) == pytest.approx(defaults[attr]), attr
    for attr in BOOL_FIELDS:
        assert _value(widgets[attr]) == defaults[attr], attr
    assert widgets["bn_grid_size_str"].currentText() == defaults["bn_grid_size_str"]

    # The shared model reflects the reset values too.
    state = window.collect_settings_state()
    assert state.bn_perc_low == defaults["bn_perc_low"]
    assert state.apply_bn is True
    assert state.apply_coverage_render is False
    assert state.cb_max_b_factor == pytest.approx(1.5)


def test_reset_expert_settings_is_gui_only(window):
    """Reset never touches the engine/settings file/preview source."""
    source_before = window._preview_source
    window.reset_expert_button.click()
    # No settings file is written (bare MainWindow has no settings path).
    assert window._settings_path is None
    # The preview source is untouched (identity holds; it is None without data).
    assert window._preview_source is source_before


# --------------------------------------------------------------------------
# (1b) Expert-tab field labels localize FR/EN
# --------------------------------------------------------------------------
def test_expert_field_labels_localize(window):
    assert localization.translate("field_apply_bn", "en") == "Enable BN"
    assert localization.translate("field_apply_bn", "fr") == "Activer BN"
    assert localization.translate("reset_expert_button", "fr") == "Réinitialiser les réglages Expert"

    label = _find_label(window, "Enable BN")
    assert label is not None

    window.language_combo.setCurrentText("Français")
    assert label.text() == "Activer BN"
    assert window.reset_expert_button.text() == "Réinitialiser les réglages Expert"
    assert window.expert_warning_label.text() == "Réglages Expert !"

    window.language_combo.setCurrentText("English")
    assert label.text() == "Enable BN"
    assert window.reset_expert_button.text() == "Reset Expert Settings"
    assert window.expert_warning_label.text() == "Expert Settings!"


def test_expert_localization_keys_have_full_parity():
    for key in (
        "field_apply_bn",
        "field_bn_grid_size",
        "field_apply_cb",
        "field_apply_final_crop",
        "field_apply_photutils_bn",
        "field_apply_batch_feathering",
        "field_apply_coverage_render",
        "field_apply_low_wht_mask",
        "reset_expert_button",
        "expert_warning_text",
    ):
        entry = localization.TRANSLATIONS[key]
        assert set(entry) == {"en", "fr"}, key
        assert entry["en"] and entry["fr"], key


# --------------------------------------------------------------------------
# (4) histogram x-range survives a refresh after manual zoom
# --------------------------------------------------------------------------
def test_histogram_manual_zoom_survives_refresh_and_resets():
    view = HistogramView()
    try:
        view.resize(256, 80)
        view.set_data(render_preview_image(_grad(256, 8, 0, 255)))
        assert view.view_range == (0.0, 1.0)

        view.zoom_histogram()
        zoomed = view.view_range
        assert zoomed[0] == 0.0 and zoomed[1] < 1.0

        # A refresh with new data must preserve the manual zoom window.
        view.set_data(render_preview_image(_grad(256, 8, 20, 200)))
        assert view.view_range == zoomed

        # reset view clears the frozen zoom -> next refresh is full range.
        view.reset_histogram_view()
        assert view.view_range == (0.0, 1.0)
        view.set_data(render_preview_image(_grad(256, 8, 0, 255)))
        assert view.view_range == (0.0, 1.0)

        # reset zoom clears it too.
        view.zoom_histogram()
        view.reset_zoom()
        assert view.view_range == (0.0, 1.0)
        view.set_data(render_preview_image(_grad(256, 8, 0, 255)))
        assert view.view_range == (0.0, 1.0)
    finally:
        view.deleteLater()


def test_histogram_auto_zoom_still_reapplies_on_refresh():
    view = HistogramView()
    try:
        view.resize(256, 80)
        view.set_data(render_preview_image(_grad(256, 8, 0, 255)))
        view.auto_zoom_enabled = True
        view.zoom_histogram()
        assert view.view_range[1] < 1.0
        # Auto-zoom re-applies on each refresh (unchanged M14 behaviour).
        view.set_data(render_preview_image(_grad(256, 8, 10, 200)))
        assert view.view_range[0] == 0.0 and view.view_range[1] < 1.0
    finally:
        view.deleteLater()


# --------------------------------------------------------------------------
# (5) dead-code removal (render_histogram_pixmap) is gone
# --------------------------------------------------------------------------
def test_render_histogram_pixmap_removed():
    import seestar.gui_qt.preview_adjust as pa

    assert not hasattr(pa, "render_histogram_pixmap")
    assert not hasattr(pa, "_HISTOGRAM_COLORS")
