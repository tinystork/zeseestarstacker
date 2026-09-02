"""M25.5-C seam tests: the Qt "System" tab (language + GPU + appearance).

Offscreen tests for the new fourth left-panel tab, which groups the
application/runtime settings instead of scattering them in the scientific
workflow:

* the System tab exists at index 2 (Stacking | Expert | System | Preview
  controls) with a translated label in both languages,
* the Language switch (moved from the left-panel top) still flips the UI
  language live and persists through the M8 save/load round-trip,
* the "Use GPU" toggle (moved from the Stacking tab) still sets the same
  ``use_gpu`` state key and reaches the stacker instance through the M20
  RunRequest/seam plumbing (unchanged),
* the Appearance theme selector defaults to System, applies Dark/Light
  immediately via a Qt palette, re-reads the platform palette on System, and
  persists through save/load,
* no ``zealfie`` token anywhere in ``gui_qt`` (source scan), and no GPU status
  label is invented (Zsss has no public GPU probe).

No real stacking, no engine, no Tk, no FITS/PNG writes, no subprocess.
``QT_QPA_PLATFORM=offscreen`` is set defensively before any ``QApplication`` is
created.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.backend_runner import (
    BackendRunResult,
    SeestarQueuedStackerBackend,
)
from seestar.gui_qt.settings_state import QtSettingsState

ROOT = Path(__file__).resolve().parents[1]


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


def _is_descendant(widget, ancestor) -> bool:
    """True if ``widget`` is ``ancestor`` or a child/grandchild of it."""
    while widget is not None:
        if widget is ancestor:
            return True
        widget = widget.parent()
    return False


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return path


# --------------------------------------------------------------------------
# System tab structure + localization
# --------------------------------------------------------------------------
def test_system_tab_exists_at_index_2_with_translated_label(window):
    assert window.tabs.count() == 4
    assert window.tabs.tabText(0) == "Stacking"
    assert window.tabs.tabText(1) == "Expert"
    assert window.tabs.tabText(2) == "System"
    assert window.tabs.tabText(3) == "Preview controls"
    assert window.tabs.widget(2) is window._system_tab

    window.language_combo.setCurrentText("Français")
    assert window.tabs.tabText(2) == "Système"
    window.language_combo.setCurrentText("English")
    assert window.tabs.tabText(2) == "System"


# --------------------------------------------------------------------------
# Language control moved into System (live + persisted)
# --------------------------------------------------------------------------
def test_language_control_lives_in_system_not_left_panel_top(window):
    # The combo still exists and is enabled, but it now lives inside the
    # System tab (its old home was the left-panel top, above the tab widget).
    assert window.language_combo.isEnabled()
    assert _is_descendant(window.language_combo, window._system_tab)
    assert _is_descendant(window.language_label, window._system_tab)


def test_language_switch_is_live(window):
    assert window._current_language_code() == "en"
    assert window.start_button.text() == "Start"
    window.language_combo.setCurrentText("Français")
    assert window._current_language_code() == "fr"
    assert window.settings_state.language == "fr"
    assert window.start_button.text() == "Démarrer"
    assert window.appearance_label.text() == "Apparence :"
    assert window.tabs.tabText(2) == "Système"
    window.language_combo.setCurrentText("English")
    assert window._current_language_code() == "en"


def test_language_persists_round_trip(tmp_path):
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
        assert win2.tabs.tabText(2) == "Système"
    finally:
        win2.shutdown()


# --------------------------------------------------------------------------
# GPU control moved into System (same state key + M20 seam unchanged)
# --------------------------------------------------------------------------
def test_gpu_control_lives_in_system_and_sets_same_state_key(window):
    assert isinstance(window.use_gpu_check, QCheckBox)
    assert _is_descendant(window.use_gpu_check, window._system_tab)
    # No duplicate on the Stacking tab (its old home).
    assert window.use_gpu_check not in window._stacking_tab.findChildren(QCheckBox)

    window.use_gpu_check.setChecked(True)
    state = window.collect_settings_state()
    assert state.use_gpu is True
    assert window.build_run_request().backend_kwargs["use_gpu"] is True


class _FakeStacker:
    """Minimal stacker double (same shape as the M20 E2E double)."""

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = dict(kwargs)
        self.align_on_disk = None
        self.progress_cb = None
        self.start_kwargs = None
        self.stop_called = False
        self._running = False

    def set_progress_callback(self, cb) -> None:
        self.progress_cb = cb

    def start_processing(self, **kwargs):
        self.start_kwargs = dict(kwargs)
        self._running = True
        return True

    def is_running(self) -> bool:
        self._running = False
        return False

    def stop(self) -> None:
        self.stop_called = True
        self._running = False


def test_gpu_toggle_reaches_stackers_instance_via_seam(window):
    """The moved toggle still traverses Qt -> state -> RunRequest -> seam."""
    instances = []

    def factory(**kwargs):
        stacker = _FakeStacker(**kwargs)
        instances.append(stacker)
        return stacker

    backend = SeestarQueuedStackerBackend(stacker_factory=factory, poll_interval=0.001)
    window.use_gpu_check.setChecked(True)
    request = window.build_run_request()
    assert request.backend_kwargs["use_gpu"] is True

    result = backend.run(request, lambda p: None, lambda m: None, lambda: False)
    assert result is BackendRunResult.FINISHED
    assert instances[0].use_gpu is True
    # Seam fields never leak into the start_processing surface.
    assert "use_gpu" not in instances[0].start_kwargs


# --------------------------------------------------------------------------
# Theme (appearance) — default, immediate palette, persistence
# --------------------------------------------------------------------------
def test_theme_default_is_system(window):
    assert window.theme_mode == "system"
    assert window.settings_state.theme == "system"
    assert window.theme_combo.currentIndex() == 0
    assert [window.theme_combo.itemText(i) for i in range(window.theme_combo.count())] == [
        "System",
        "Dark",
        "Light",
    ]


def test_theme_dark_light_applies_palette_and_system_restores(window):
    app = QApplication.instance()
    system_color = app.palette().color(QPalette.ColorRole.Window)

    window.theme_combo.setCurrentIndex(1)  # Dark
    assert window.theme_mode == "dark"
    assert window.settings_state.theme == "dark"
    dark_color = app.palette().color(QPalette.ColorRole.Window)
    assert dark_color != system_color
    assert dark_color == QColor(53, 53, 53)

    window.theme_combo.setCurrentIndex(2)  # Light
    assert window.theme_mode == "light"
    light_color = app.palette().color(QPalette.ColorRole.Window)
    assert light_color != dark_color
    assert light_color != system_color

    window.theme_combo.setCurrentIndex(0)  # System -> re-read platform palette
    assert window.theme_mode == "system"
    assert app.palette().color(QPalette.ColorRole.Window) == system_color


def test_theme_localizes_fr_en(window):
    window.language_combo.setCurrentText("Français")
    assert [window.theme_combo.itemText(i) for i in range(window.theme_combo.count())] == [
        "Système",
        "Sombre",
        "Clair",
    ]
    window.language_combo.setCurrentText("English")
    assert [window.theme_combo.itemText(i) for i in range(window.theme_combo.count())] == [
        "System",
        "Dark",
        "Light",
    ]


def test_theme_persists_round_trip(tmp_path):
    p = str(tmp_path / "seestar_settings.json")
    win = MainWindow(settings_path=p)
    win.theme_combo.setCurrentIndex(1)  # Dark
    assert win.theme_mode == "dark"
    win.shutdown()

    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["theme"] == "dark"

    win2 = MainWindow(settings_path=p)
    try:
        assert win2.theme_mode == "dark"
        assert win2.theme_combo.currentIndex() == 1
        assert (
            QApplication.instance().palette().color(QPalette.ColorRole.Window)
            == QColor(53, 53, 53)
        )
    finally:
        win2.shutdown()


def test_theme_state_round_trip_model():
    state = QtSettingsState()
    assert state.theme == "system"
    state.theme = "dark"
    assert QtSettingsState.from_dict(state.to_dict()).theme == "dark"
    # Unknown persisted theme degrades to the platform default.
    assert QtSettingsState.from_dict({"theme": "neon"}).theme == "system"
    assert QtSettingsState.from_dict({"theme": None}).theme == "system"


# --------------------------------------------------------------------------
# FRP-L1 — Disabled palette group: complete and dimmed (dark + light)
# --------------------------------------------------------------------------

_DISABLED_TEXT_ROLES = (
    QPalette.ColorRole.WindowText,
    QPalette.ColorRole.Text,
    QPalette.ColorRole.ButtonText,
)

_STRUCTURAL_MIRROR_ROLES = (
    QPalette.ColorRole.Window,
    QPalette.ColorRole.Base,
    QPalette.ColorRole.AlternateBase,
    QPalette.ColorRole.Button,
    QPalette.ColorRole.ToolTipBase,
    QPalette.ColorRole.ToolTipText,
    QPalette.ColorRole.Highlight,
    QPalette.ColorRole.HighlightedText,
)

_THEME_WINDOWS = {
    "dark": QColor(53, 53, 53),
    "light": QColor(240, 240, 240),
}


def _relative_luminance(color: QColor) -> float:
    """WCAG relative luminance of an sRGB colour (0..1)."""
    def _lin(channel: int) -> float:
        v = channel / 255.0
        if v <= 0.03928:
            return v / 12.92
        return ((v + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * _lin(color.red())
        + 0.7152 * _lin(color.green())
        + 0.0722 * _lin(color.blue())
    )


def _contrast_ratio(a: QColor, b: QColor) -> float:
    """WCAG contrast ratio between two colours (>= 1.0)."""
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _theme_palettes(app, window):
    """Yield ``(mode, palette)`` after applying Dark then Light."""
    for idx, mode in ((1, "dark"), (2, "light")):
        window.theme_combo.setCurrentIndex(idx)
        assert window.theme_mode == mode
        yield mode, app.palette()


def test_theme_disabled_windowtext_dimmed_and_readable(window):
    """FRP-L1: Disabled ``WindowText`` must never fall back to the Active
    full-contrast color (black-on-white / white-on-dark is the release-blocking
    readability defect) — it is dimmed, distinct from Active, and stays
    readable (>= 3.0:1) against the theme ``Window`` in both dark and light.
    The Active ``Window`` colors themselves stay byte-identical."""
    app = QApplication.instance()
    for mode, palette in _theme_palettes(app, window):
        active_wt = palette.color(
            QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText
        )
        disabled_wt = palette.color(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText
        )
        bg = palette.color(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Window
        )
        # The defect: disabled == active full-contrast text.  Fixed: dimmed.
        assert disabled_wt != active_wt, mode
        assert _contrast_ratio(disabled_wt, bg) >= 3.0, (
            mode, disabled_wt.name(), bg.name(),
            _contrast_ratio(disabled_wt, bg),
        )
        # Active-group guard: the normal appearance is unchanged.
        assert palette.color(
            QPalette.ColorGroup.Active, QPalette.ColorRole.Window
        ) == _THEME_WINDOWS[mode], mode


def test_theme_disabled_text_and_buttontext_dimmed_and_readable(window):
    """FRP-L1: Disabled ``Text`` / ``ButtonText`` are dimmed (distinct from
    their Active values) and readable (>= 3.0:1) against ``Base`` / ``Button``
    in both themes (light at minimum)."""
    app = QApplication.instance()
    pairs = (
        (QPalette.ColorRole.Text, QPalette.ColorRole.Base),
        (QPalette.ColorRole.ButtonText, QPalette.ColorRole.Button),
    )
    for mode, palette in _theme_palettes(app, window):
        for text_role, bg_role in pairs:
            active = palette.color(QPalette.ColorGroup.Active, text_role)
            disabled = palette.color(QPalette.ColorGroup.Disabled, text_role)
            bg = palette.color(QPalette.ColorGroup.Disabled, bg_role)
            assert disabled != active, (mode, text_role)
            assert _contrast_ratio(disabled, bg) >= 3.0, (
                mode, text_role, disabled.name(), bg.name(),
                _contrast_ratio(disabled, bg),
            )


def test_theme_disabled_structural_roles_mirror_active(window):
    """FRP-L1: the Disabled STRUCTURAL roles mirror their Active values in
    both themes (a disabled surface never falls back to a platform/style
    default that clashes with the theme), and the Disabled text roles are
    dimmed away from the Active full-contrast values."""
    app = QApplication.instance()
    for mode, palette in _theme_palettes(app, window):
        for role in _STRUCTURAL_MIRROR_ROLES:
            assert palette.color(
                QPalette.ColorGroup.Disabled, role
            ) == palette.color(QPalette.ColorGroup.Active, role), (mode, role)
        for role in _DISABLED_TEXT_ROLES:
            assert palette.color(
                QPalette.ColorGroup.Disabled, role
            ) != palette.color(QPalette.ColorGroup.Active, role), (mode, role)


# --------------------------------------------------------------------------
# Import hygiene / GPU status label decision
# --------------------------------------------------------------------------
def test_no_zealfie_token_anywhere_in_gui_qt_source():
    """Hard invariant: no ``zealfie`` token in any gui_qt source file."""
    import seestar.gui_qt as gui_qt

    pkg_dir = Path(gui_qt.__file__).resolve().parent
    for py in pkg_dir.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert "zealfie" not in text, f"{py.name} references zealfie"


def test_no_gpu_status_label_invented(window):
    """Zsss has no public GPU probe, so no status label is invented.

    The decision (documented in the checklist) is to omit the label entirely
    rather than invent a probe that imports the engine/ZeAlfie or adds a
    dependency.  Asserting no such widget exists pins that decision.
    """
    assert not hasattr(window, "gpu_status_label")
    assert not hasattr(window, "gpu_status")
