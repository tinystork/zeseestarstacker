"""M9 seam tests: Qt FR/EN language switch + Qt-local localization surface.

Offscreen tests for the bounded, user-triggered FR/EN switch in the Qt shell:

* the language combo is enabled and defaults to English,
* switching to French (and back) updates ``_current_language_code()`` and the
  representative visible labels without rebuilding the window,
* the ZeAnalyser launch seam passes the selected language as ``--lang``,
* the persisted ``language`` field round-trips through the M8 settings JSON and
  an unknown/corrupt value falls back to English,
* the Qt-local ``seestar.gui_qt.localization`` mapping has full key parity
  (every key carries both ``en`` and ``fr``) and never crashes on a missing key,
* ``import seestar.gui_qt`` stays free of Tk / engine / private solver imports.

``QT_QPA_PLATFORM=offscreen`` is set defensively before any ``QApplication`` is
created.  No real stacking, no engine, no Tk.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QGroupBox

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt import localization
from seestar.gui_qt.settings_state import QtSettingsState


@pytest.fixture(scope="session", autouse=True)
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


@pytest.fixture()
def window(qapp):
    win = MainWindow()
    yield win
    win.shutdown()


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return path


def _settings_group_titles(win):
    return sorted(g.title() for g in win._settings_tab.findChildren(QGroupBox))


# --------------------------------------------------------------------------
# Pure Qt-local localization module (key parity + fallback)
# --------------------------------------------------------------------------
def test_translation_key_parity():
    assert localization.SUPPORTED_LANGUAGES == ("en", "fr")
    assert localization.LANGUAGE_CODE_BY_TEXT == {"English": "en", "Français": "fr"}
    for key, entry in localization.TRANSLATIONS.items():
        assert set(entry) == {"en", "fr"}, f"key {key!r} must have en+fr"
        assert entry["en"], f"key {key!r} has empty en"
        assert entry["fr"], f"key {key!r} has empty fr"


def test_translate_and_fallback_never_raise():
    assert localization.translate("start", "en") == "Start"
    assert localization.translate("start", "fr") == "Démarrer"
    # Missing key -> default, then the key itself.
    assert localization.translate("no_such_key", "fr") == "no_such_key"
    assert localization.translate("no_such_key", "fr", default="X") == "X"
    # Unsupported language -> English.
    assert localization.translate("start", "de") == "Start"
    assert localization.translate("start", None) == "Start"


def test_normalize_language_and_label_for():
    assert localization.normalize_language("en") == "en"
    assert localization.normalize_language("fr") == "fr"
    assert localization.normalize_language("de") == "en"
    assert localization.normalize_language(None) == "en"
    assert localization.normalize_language(123) == "en"
    assert localization.language_label_for("fr") == "Français"
    assert localization.language_label_for("de") == "English"
    assert localization.supported_language_codes() == ["en", "fr"]


def test_localization_module_is_pure_stdlib():
    from pathlib import Path

    import seestar.gui_qt as gui_qt

    pkg_dir = Path(gui_qt.__file__).resolve().parent
    text = (pkg_dir / "localization.py").read_text(encoding="utf-8")
    forbidden = (
        "tkinter",
        "seestar.core",
        "seestar.alignment",
        "seestar.enhancement",
        "seestar.queuep",
        "seestar.gui.settings",
        "seestar.gui.main_window",
        "zesolver_adapter",
        "zesolver.api",
        "zealfie",
        "PySide6",
        "QtCore",
        "QtGui",
        "QtWidgets",
        "import numpy",
    )
    for token in forbidden:
        assert token not in text, f"localization.py references {token}"


# --------------------------------------------------------------------------
# Combo enablement + default English
# --------------------------------------------------------------------------
def test_language_combo_enabled_and_defaults_to_english(window):
    assert window.language_combo.isEnabled()
    assert [window.language_combo.itemText(i) for i in range(window.language_combo.count())] == [
        "English",
        "Français",
    ]
    assert window.language_combo.currentText() == "English"
    assert window._current_language_code() == "en"
    assert window.start_button.text() == "Start"


def test_switch_language_updates_code_and_representative_labels(window):
    assert window.tabs.tabText(0) == "Stacking"
    assert window.tabs.tabText(1) == "Expert"
    assert window.tabs.tabText(2) == "System"
    assert window.tabs.tabText(3) == "Preview controls"
    assert window.start_button.text() == "Start"
    assert window.analyse_button.text() == "Analyse"

    window.language_combo.setCurrentText("Français")

    assert window._current_language_code() == "fr"
    assert window.settings_state.language == "fr"
    assert window.tabs.tabText(0) == "Empilement"
    assert window.tabs.tabText(1) == "Expert"
    assert window.tabs.tabText(2) == "Système"
    assert window.tabs.tabText(3) == "Contrôles d'aperçu"
    assert window.start_button.text() == "Démarrer"
    assert window.stop_button.text() == "Arrêter"
    assert window.analyse_button.text() == "Analyser"
    assert window.solver_button.text() == "Solveur"
    assert window.view_inputs_button.text() == "Voir les entrées"
    assert window.add_folder_button.text() == "Ajouter un dossier"
    assert window.open_output_button.text() == "Ouvrir la sortie"
    assert window.copy_log_button.text() == "Copier le journal"
    assert window.progress_label.text() == "Progression :"
    assert window.log_label.text() == "Journal :"
    assert window.elapsed_label.text() == "Écoulé : 0:00"
    assert window.remaining_label.text() == "Restant : —"
    assert window.preview_label.text() == "Aperçu : —"
    assert window.browse_input_button.text() == "Parcourir..."
    assert "simulé (dev/test)" in window.backend_notice_label.text()

    titles = _settings_group_titles(window)
    assert "Empilement / Chemins" in titles
    assert "Pondération par qualité" in titles
    assert "Mosaïque" in titles

    # Switching back restores English.
    window.language_combo.setCurrentText("English")
    assert window._current_language_code() == "en"
    assert window.settings_state.language == "en"
    assert window.tabs.tabText(0) == "Stacking"
    assert window.start_button.text() == "Start"
    assert window.analyse_button.text() == "Analyse"
    assert window.progress_label.text() == "Progression:"


# --------------------------------------------------------------------------
# ZeAnalyser launch uses the selected language
# --------------------------------------------------------------------------
def test_analyse_launch_uses_selected_french_language(window, tmp_path, monkeypatch):
    window.input_edit.setText(str(tmp_path))
    window._sync_state_from_controls()

    calls = {}
    monkeypatch.setattr(window, "_analyzer_command_file_maker", lambda: "/tmp/cmd.txt")

    def fake_launcher(input_folder, lang, command_file_path):
        calls["lang"] = lang
        return True

    monkeypatch.setattr(window, "_analyzer_launcher", fake_launcher)

    window.language_combo.setCurrentText("Français")
    window._on_analyse()

    assert calls["lang"] == "fr"


# --------------------------------------------------------------------------
# Persistence (M8 path) round-trip + invalid fallback
# --------------------------------------------------------------------------
def test_persisted_language_fr_loads_and_saves(tmp_path):
    p = str(tmp_path / "seestar_settings.json")
    win = MainWindow(settings_path=p)
    win.language_combo.setCurrentText("Français")
    win.shutdown()

    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["language"] == "fr"

    win2 = MainWindow(settings_path=p)
    try:
        assert win2._current_language_code() == "fr"
        assert win2.language_combo.currentText() == "Français"
        assert win2.start_button.text() == "Démarrer"
        assert win2.tabs.tabText(0) == "Empilement"
    finally:
        win2.shutdown()


def test_invalid_persisted_language_falls_back_to_english(tmp_path):
    p = str(tmp_path / "seestar_settings.json")
    _write_json(p, {"language": "de"})
    win = MainWindow(settings_path=p)
    try:
        assert win._current_language_code() == "en"
        assert win.language_combo.currentText() == "English"
        assert win.start_button.text() == "Start"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Model-level language field (settings_state)
# --------------------------------------------------------------------------
def test_settings_state_language_field():
    assert QtSettingsState().language == "en"
    assert QtSettingsState.defaults()["language"] == "en"
    assert QtSettingsState.from_dict({"language": "fr"}).language == "fr"
    assert QtSettingsState.from_dict({"language": "fr"}).to_dict()["language"] == "fr"


def test_settings_state_invalid_language_falls_back():
    assert QtSettingsState.from_dict({"language": "de"}).language == "en"
    assert QtSettingsState.from_dict({"language": 123}).language == "en"
    assert QtSettingsState.from_dict({"language": None}).language == "en"
    assert QtSettingsState.from_dict({}).language == "en"


# --------------------------------------------------------------------------
# M10: preview-controls tab labels localize (WB / stretch / histogram)
# --------------------------------------------------------------------------
def test_preview_controls_labels_localize(window):
    assert window.wb_group.title() == "White balance"
    assert window.stretch_group.title() == "Stretch"
    assert window.wb_reset_button.text() == "Reset"
    assert window.auto_stretch_button.text() == "Auto Stretch"

    window.language_combo.setCurrentText("Français")

    assert window.wb_group.title() == "Balance des blancs"
    assert window.stretch_group.title() == "Étirement"
    assert window.wb_reset_button.text() == "Réinitialiser"
    assert window.auto_stretch_button.text() == "Étirement auto"

    # Round-trip back to English.
    window.language_combo.setCurrentText("English")
    assert window.wb_group.title() == "White balance"
    assert window.auto_stretch_button.text() == "Auto Stretch"


def test_right_panel_histogram_labels_localize(window):
    assert window.right_histogram_group.title() == "Histogram"
    assert window.right_histogram_status.text() == "No preview"
    assert window.auto_zoom_histo_check.text() == "Auto zoom histogram"
    assert window.hist_reset_view_button.text() == "Reset Histogram"
    assert window.hist_zoom_button.text() == "Zoom Histogram"

    window.language_combo.setCurrentText("Français")

    assert window.right_histogram_group.title() == "Histogramme"
    assert window.right_histogram_status.text() == "Aucun aperçu"
    assert window.auto_zoom_histo_check.text() == "Zoom auto histogramme"
    assert window.hist_reset_view_button.text() == "Réinitialiser l'histogramme"
    assert window.hist_zoom_button.text() == "Zoom histogramme"

    # Round-trip back to English.
    window.language_combo.setCurrentText("English")
    assert window.right_histogram_group.title() == "Histogram"
    assert window.right_histogram_status.text() == "No preview"
    assert window.auto_zoom_histo_check.text() == "Auto zoom histogram"
