"""Action-band ergonomics tests for the Qt shell (M25.5-E).

Covers the layout-only change that turns the right-panel "Actions" QGroupBox
from a dense 2-column QGridLayout (with ``open_output_button`` spanning the
full width) into a compact, airy single-band QHBoxLayout mirroring the Tk
right-panel ``control_frame`` (run controls left, folder actions right, no
full-width buttons, Start emphasized as the default button).

Explicitly *not* covered / guaranteed here (because the change is layout-only):

* no behaviour change — the same signals, enablement rules, state keys and
  translations are asserted to be untouched;
* no stylesheet — Start is distinguished only via ``setDefault(True)``;
* no Tk/engine/FITS/PNG writes and no subprocess.

``QT_QPA_PLATFORM=offscreen`` is set defensively before any ``QApplication``
is created, mirroring the other Qt shell tests.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QGroupBox, QHBoxLayout, QSizePolicy

from seestar.gui_qt import MainWindow, create_application

ACTION_BUTTON_ATTRS = (
    "start_button",
    "stop_button",
    "analyse_button",
    "solver_button",
    "view_inputs_button",
    "add_folder_button",
    "open_output_button",
)


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Single process-wide QApplication for the whole session."""
    app = create_application([])
    assert app is QApplication.instance()
    return app


@pytest.fixture()
def window(qapp):
    win = MainWindow()
    yield win
    win.shutdown()


def _action_buttons(win):
    return [getattr(win, attr) for attr in ACTION_BUTTON_ATTRS]


# --------------------------------------------------------------------------
# Presence + parent group (same keys, same group, no grid)
# --------------------------------------------------------------------------
def test_all_action_buttons_exist_with_same_keys(window):
    for attr in ACTION_BUTTON_ATTRS:
        assert hasattr(window, attr), f"missing action button {attr}"


def test_action_buttons_share_the_actions_group_parent(window):
    assert isinstance(window.actions_group, QGroupBox)
    for button in _action_buttons(window):
        assert button.parent() is window.actions_group, (
            f"{button.text()!r} is not parented to the actions group"
        )


def test_actions_group_is_a_single_band_not_a_grid(window):
    # M25.5-E: the group now hosts a QHBoxLayout, never a QGridLayout — this is
    # what removes the 2-column cells and the full-width Open Output span.
    assert isinstance(window.actions_group.layout(), QHBoxLayout)


def test_no_action_button_has_expanding_horizontal_policy(window):
    # Offscreen-safe "no full-width button" proof: none of the action buttons
    # is allowed to expand horizontally to fill the band.
    for button in _action_buttons(window):
        assert button.sizePolicy().horizontalPolicy() != QSizePolicy.Policy.Expanding, (
            f"{button.text()!r} has an Expanding horizontal policy"
        )


def test_open_output_lost_its_full_width_span(window):
    # In a QHBoxLayout every button is a single cell of its own; Open Output is
    # no longer a 2-column spanning widget.  It is simply the last button.
    layout = window.actions_group.layout()
    items = [layout.itemAt(i) for i in range(layout.count())]
    widgets = [it.widget() for it in items if it.widget() is not None]
    assert widgets[-1] is window.open_output_button
    # A stretch sits between the run controls and the folder actions, so no
    # widget is stretched across the full band width.
    assert any(it.spacerItem() is not None for it in items)


def test_start_is_visually_distinct_as_default_button(window):
    assert window.start_button.isDefault()
    for button in _action_buttons(window):
        if button is not window.start_button:
            assert not button.isDefault(), (
                f"{button.text()!r} must not be the default button"
            )


# --------------------------------------------------------------------------
# Layout sanity: visibility + panel placement + backend notice below
# --------------------------------------------------------------------------
def test_all_action_buttons_visible_after_show(qapp, window):
    window.show()
    qapp.processEvents()
    try:
        for button in _action_buttons(window):
            assert button.isVisible(), f"{button.text()!r} not visible"
    finally:
        window.hide()


def test_actions_group_lives_in_the_right_panel(window):
    # The action band is part of the persistent right preview/action panel
    # (Tk ``control_frame`` parity), not the left scroll area.
    assert window.actions_group.parent() is window.right_panel


def test_backend_notice_stays_below_actions_group(window):
    layout = window.right_panel.layout()
    assert layout.indexOf(window.actions_group) < layout.indexOf(window.backend_notice_label)


# --------------------------------------------------------------------------
# Enablement / wiring is unchanged
# --------------------------------------------------------------------------
def test_start_stop_run_state_gating(window):
    # Fresh window: Start enabled, Stop disabled (same pre-M25.5-E contract).
    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()

    window._running = True
    window._update_run_state()
    assert not window.start_button.isEnabled()
    assert window.stop_button.isEnabled()

    window._running = False
    window._update_run_state()
    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()


def test_analyse_disabled_at_construction(window):
    assert not window.analyse_button.isEnabled()


def test_analyse_enabled_for_existing_input(window, tmp_path):
    window.input_edit.setText(str(tmp_path))
    window._sync_state_from_controls()
    assert window.analyse_button.isEnabled()


def test_path_action_enablement_driven_by_path_state(window, tmp_path):
    # Empty paths -> all path actions disabled.
    window._sync_state_from_controls()
    assert not window.view_inputs_button.isEnabled()
    assert not window.open_output_button.isEnabled()
    assert not window.add_folder_button.isEnabled()

    window.input_edit.setText(str(tmp_path))
    window.output_edit.setText(str(tmp_path))
    window._sync_state_from_controls()
    assert window.view_inputs_button.isEnabled()
    assert window.open_output_button.isEnabled()
    assert window.add_folder_button.isEnabled()


def test_add_folder_disabled_while_running(window, tmp_path):
    window.input_edit.setText(str(tmp_path))
    window._sync_state_from_controls()
    assert window.add_folder_button.isEnabled()

    window._running = True
    window._update_run_state()
    assert not window.add_folder_button.isEnabled()
    assert window.view_inputs_button.isEnabled()  # still viewable while running


# --------------------------------------------------------------------------
# Translations still switch live (FR/EN)
# --------------------------------------------------------------------------
def test_action_band_translates_live(window):
    assert window.actions_group.title() == "Actions"
    assert window.start_button.text() == "Start"
    assert window.analyse_button.text() == "Analyse"

    window.language_combo.setCurrentText("Français")

    assert window.actions_group.title() == "Actions"  # identical in fr
    assert window.start_button.text() == "Démarrer"
    assert window.stop_button.text() == "Arrêter"
    assert window.analyse_button.text() == "Analyser"
    assert window.solver_button.text() == "Solveur"
    assert window.view_inputs_button.text() == "Voir les entrées"
    assert window.add_folder_button.text() == "Ajouter un dossier"
    assert window.open_output_button.text() == "Ouvrir la sortie"

    window.language_combo.setCurrentText("English")
    assert window.start_button.text() == "Start"
    assert window.analyse_button.text() == "Analyse"
