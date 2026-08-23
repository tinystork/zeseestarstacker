"""M10 seam tests: the functional, grouped Qt Settings surface.

These tests exercise the replacement of the former ``Settings placeholder`` tab
with a scrollable, sectioned, Qt-native settings surface that mirrors the
backend-relevant :class:`~seestar.gui_qt.settings_state.QtSettingsState` fields
not already shown on the Stack tab.

They verify, under the ``offscreen`` platform plugin:

* the Settings tab is no longer the placeholder text and has real sections and
  controls,
* representative bool/int/float/string/combo/list/tri-state and nested-mosaic
  controls mirror into ``QtSettingsState`` *and*
  ``build_run_request().backend_kwargs``,
* the Stack tab controls keep mirroring as before (no regression),
* the simulated-backend notice is present in default mode and differs in
  ``backend_mode="seestar"``,
* ``import seestar.gui_qt`` stays free of numpy/PIL/matplotlib/Tk/engine.

No real stacking is performed; the Settings tab never touches the engine.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QWidget,
)

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.main_window import (
    SEESTAR_BACKEND_NOTICE,
    SIMULATED_BACKEND_NOTICE,
    TAB_EXPERT,
)
from seestar.gui_qt.settings_state import QtSettingsState

ROOT = Path(__file__).resolve().parents[1]


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


def _settings_tab_widget(win) -> QWidget:
    for i in range(win.tabs.count()):
        if win.tabs.tabText(i) == TAB_EXPERT:
            return win.tabs.widget(i)
    raise AssertionError("no Expert tab found")


# --------------------------------------------------------------------------
# Structure: the tab is no longer a placeholder
# --------------------------------------------------------------------------
def test_settings_tab_is_not_placeholder(window):
    tab = _settings_tab_widget(window)
    assert isinstance(tab, QScrollArea)
    labels = tab.findChildren(QLabel)
    texts = [lbl.text() for lbl in labels]
    assert "Settings placeholder" not in texts
    groups = tab.findChildren(QGroupBox)
    assert len(groups) >= 11, f"expected grouped sections, got {len(groups)}"


def test_drizzle_advanced_section_removed_from_expert(window):
    """The four drizzle sub-options moved to the Stacking tab (D3), so the
    Expert surface must no longer expose a "Drizzle Advanced" group or the
    drizzle_scale / drizzle_wht_threshold / drizzle_kernel / drizzle_pixfrac
    widget keys."""
    tab = _settings_tab_widget(window)
    titles = [g.title() for g in tab.findChildren(QGroupBox)]
    assert "Drizzle Advanced" not in titles
    for attr in ("drizzle_scale", "drizzle_wht_threshold", "drizzle_kernel", "drizzle_pixfrac"):
        assert attr not in window._settings_widgets, attr


def test_settings_surface_has_real_controls(window):
    assert isinstance(window._settings_widgets, dict)
    assert len(window._settings_widgets) >= 58
    assert len(window._mosaic_widgets) == 14


def test_representative_control_per_section(window):
    """At least one backend-relevant control exists for every section group."""
    expected = {
        "kappa": QDoubleSpinBox,                  # Stacking / Paths
        "correct_hot_pixels": QCheckBox,          # Calibration / Hot Pixels
        "weight_by_snr": QCheckBox,               # Quality Weighting
        "apply_chroma_correction": QCheckBox,     # Colour / Post-processing
        "apply_master_tile_crop": QCheckBox,      # Cropping
        "apply_photutils_bn": QCheckBox,          # Photutils BN
        "apply_feathering": QCheckBox,            # Feathering / Low-weight Mask
        "mosaic_mode_active": QCheckBox,          # Mosaic
        "astap_path": QLineEdit,                  # Solver
        "save_final_as_float32": QCheckBox,       # Output / Reprojection
        "match_background_for_final": QComboBox,  # Final Background Matching
    }
    for attr, wtype in expected.items():
        assert attr in window._settings_widgets, f"missing control {attr}"
        assert isinstance(window._settings_widgets[attr], wtype), attr


def test_widget_types_match_field_kinds(window):
    widgets = window._settings_widgets
    assert isinstance(widgets["correct_hot_pixels"], QCheckBox)
    assert isinstance(widgets["neighborhood_size"], QSpinBox)
    assert isinstance(widgets["snr_exponent"], QDoubleSpinBox)
    assert isinstance(widgets["astap_path"], QLineEdit)
    assert isinstance(widgets["stack_norm_method"], QComboBox)
    assert isinstance(widgets["order_file_list"], QLineEdit)
    assert isinstance(widgets["match_background_for_final"], QComboBox)
    assert isinstance(window._mosaic_widgets["kernel"][1], QComboBox)


# --------------------------------------------------------------------------
# Mirroring: widget -> QtSettingsState -> backend_kwargs (one per category)
# --------------------------------------------------------------------------
def test_bool_field_updates_state_and_kwargs(window):
    window._settings_widgets["correct_hot_pixels"].setChecked(False)
    state = window.collect_settings_state()
    assert state.correct_hot_pixels is False
    kw = window.build_run_request().backend_kwargs
    assert kw["correct_hot_pixels"] is False


def test_int_field_updates_state_and_kwargs(window):
    window._settings_widgets["neighborhood_size"].setValue(7)
    state = window.collect_settings_state()
    assert state.neighborhood_size == 7
    kw = window.build_run_request().backend_kwargs
    assert kw["neighborhood_size"] == 7


def test_float_field_updates_state_and_kwargs(window):
    window._settings_widgets["snr_exponent"].setValue(2.5)
    state = window.collect_settings_state()
    assert state.snr_exponent == pytest.approx(2.5)
    kw = window.build_run_request().backend_kwargs
    assert kw["snr_exp"] == pytest.approx(2.5)


def test_string_field_updates_state_and_kwargs(window):
    window._settings_widgets["astap_path"].setText("/usr/bin/astap")
    state = window.collect_settings_state()
    assert state.astap_path == "/usr/bin/astap"
    kw = window.build_run_request().backend_kwargs
    assert kw["astap_path"] == "/usr/bin/astap"


def test_combo_field_updates_state_and_kwargs(window):
    window._settings_widgets["stack_norm_method"].setCurrentText("linear_fit")
    state = window.collect_settings_state()
    assert state.stack_norm_method == "linear_fit"
    kw = window.build_run_request().backend_kwargs
    assert kw["normalize_method"] == "linear_fit"


def test_list_field_round_trips_into_kwargs(window):
    window._settings_widgets["order_file_list"].setText("a.fit, b.fit, c.fit")
    state = window.collect_settings_state()
    assert state.order_file_list == ["a.fit", "b.fit", "c.fit"]
    kw = window.build_run_request().backend_kwargs
    assert kw["ordered_files"] == ["a.fit", "b.fit", "c.fit"]


def test_match_background_tristate_semantics(window):
    combo = window._settings_widgets["match_background_for_final"]
    # Default state -> None.
    state = window.collect_settings_state()
    assert state.match_background_for_final is None
    assert window.build_run_request().backend_kwargs["match_background_for_final"] is None

    combo.setCurrentText("true")
    state = window.collect_settings_state()
    assert state.match_background_for_final is True
    assert window.build_run_request().backend_kwargs["match_background_for_final"] is True

    combo.setCurrentText("false")
    state = window.collect_settings_state()
    assert state.match_background_for_final is False
    assert window.build_run_request().backend_kwargs["match_background_for_final"] is False

    combo.setCurrentText("default")
    state = window.collect_settings_state()
    assert state.match_background_for_final is None
    assert window.build_run_request().backend_kwargs["match_background_for_final"] is None


def test_mosaic_active_and_nested_dict_update_kwargs(window):
    window.mosaic_active_check.setChecked(True)
    kernel_combo = window._mosaic_widgets["kernel"][1]
    kernel_combo.setCurrentText("gaussian")
    window._mosaic_widgets["fastalign_orb_features"][1].setValue(5000)

    state = window.collect_settings_state()
    assert state.mosaic_mode_active is True
    assert state.mosaic_settings["kernel"] == "gaussian"
    assert state.mosaic_settings["fastalign_orb_features"] == 5000

    kw = window.build_run_request().backend_kwargs
    assert kw["is_mosaic_run"] is True
    assert kw["mosaic_settings"]["kernel"] == "gaussian"
    assert kw["mosaic_settings"]["fastalign_orb_features"] == 5000


def test_settings_defaults_seed_from_model(window):
    """Initial widgets are seeded from the QtSettingsState defaults."""
    state = window.collect_settings_state()
    assert state.correct_hot_pixels is True
    assert state.neighborhood_size == 5
    assert state.snr_exponent == pytest.approx(1.8)
    assert state.astap_path == ""
    assert state.stack_norm_method == "none"
    assert state.match_background_for_final is None
    assert state.order_file_list == []
    assert state.mosaic_settings["kernel"] == "square"


# --------------------------------------------------------------------------
# Stack tab controls still mirror (no regression)
# --------------------------------------------------------------------------
def test_stack_tab_controls_still_mirror(window):
    window.input_edit.setText("/in")
    window.batch_spin.setValue(9)
    window.stacking_mode_combo.setCurrentText("median")
    window.drizzle_check.setChecked(True)
    window.drizzle_group_spin.setValue(33)

    state = window.collect_settings_state()
    assert state.input_folder == "/in"
    assert state.batch_size == 9
    assert state.stacking_mode == "median"
    assert state.use_drizzle is True
    assert state.drizzle_group_size == 33

    kw = window.build_run_request().backend_kwargs
    assert kw["input_dir"] == "/in"
    assert kw["batch_size"] == 9
    assert kw["stacking_mode"] == "median"
    assert kw["use_drizzle"] is True
    assert kw["drizzle_group_size"] == 33


def test_stack_and_settings_tabs_share_same_state(window):
    """Both tabs feed the single QtSettingsState model."""
    window.batch_spin.setValue(6)
    window._settings_widgets["kappa"].setValue(3.5)
    state = window.collect_settings_state()
    assert state.batch_size == 6
    assert state.kappa == pytest.approx(3.5)
    assert state is window.settings_state


# --------------------------------------------------------------------------
# Backend-mode notice
# --------------------------------------------------------------------------
def test_simulated_backend_notice_default(window):
    assert window.backend_notice_label is not None
    text = window.backend_notice_label.text()
    assert text == SIMULATED_BACKEND_NOTICE
    assert "simulated" in text.lower()
    assert "dev/test" in text
    assert "--backend seestar" in text
    assert not window.backend_notice_label.isHidden()


def test_seestar_backend_notice_differs(qapp):
    win = MainWindow(backend_mode="seestar")
    try:
        text = win.backend_notice_label.text()
        assert text == SEESTAR_BACKEND_NOTICE
        assert text != SIMULATED_BACKEND_NOTICE
        assert "seestar" in text.lower()
        assert "--backend seestar" not in text
        win.language_combo.setCurrentText("Français")
        assert win.backend_notice_label.text() == "Moteur : seestar — traitement réel."
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Import hygiene (fresh interpreter)
# --------------------------------------------------------------------------
def test_import_hygiene_no_numpy_pil_matplotlib():
    code = (
        "import sys\n"
        "import seestar.gui_qt  # noqa: F401\n"
        "_bad = [m for m in sys.modules\n"
        "        if m.split('.')[0] in ('numpy', 'PIL', 'matplotlib', 'tkinter')\n"
        "        or m.startswith('seestar.core')\n"
        "        or m.startswith('seestar.alignment')\n"
        "        or m.startswith('seestar.enhancement')\n"
        "        or m.startswith('seestar.queuep')\n"
        "        or m in ('seestar.gui.main_window', 'seestar.gui.settings')]\n"
        "if _bad:\n"
        "    print('BAD_MODULES:', _bad)\n"
        "    sys.exit(1)\n"
        "print('IMPORT_HYGIENE_OK')\n"
    )
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert proc.returncode == 0, (
        f"import hygiene violated: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
