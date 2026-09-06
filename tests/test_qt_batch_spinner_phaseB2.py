"""Phase B2 (canonical batch contract — Qt UX) focused suite.

Mirrors mission §14 / §15 / §16 on a real offscreen ``MainWindow``:

* the batch spinner is canonical ``0 = Auto``: range ``0 .. supported_max``
  (no ``-1``), localized EN/FR tooltip, initial value 0;
* the spinner is NEVER disabled: ``0 -> 1 -> 2 -> 3`` and ``3 -> 2 -> 1 -> 0``
  both navigate freely and passing through 1 (Boring) cannot trap the
  control — the boring checkbox is a second view of the same state
  (checked <=> ``batch_size == 1``), reconciled through an in-flight guard,
  not by disabling widgets;
* boring gating of incompatible settings is NON-DESTRUCTIVE: a requested
  Drizzle state (and every drizzle sub-option value) survives a transient
  0 -> 1 -> 2 pass-through and is restored when boring mode is left;
* a legacy persisted ``-1`` loads as canonical Auto ``0``; persisted ``0``
  stays Auto ``0``;
* Boring subprocess launch routes ONLY on the effective batch mode
  (``batch_size == 1`` at Start), never because the spinner transiently
  crossed 1.

No stacking, no engine science, no FITS.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt import localization
from seestar.gui_qt.settings_state import QtSettingsState

# Spinbox supported maximum (UI cap for explicit batch capacities).
SUPPORTED_MAX = 1_000_000


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


# --------------------------------------------------------------------------
# 0 = Auto (range / initial / tooltip)
# --------------------------------------------------------------------------
def test_initial_zero_is_auto(window):
    assert window.batch_spin.minimum() == 0
    assert window.batch_spin.maximum() == SUPPORTED_MAX
    assert window.batch_spin.value() == 0
    assert window.batch_spin.isEnabled()
    assert not window.boring_check.isChecked()
    # The transmitted model value is the canonical Auto 0.
    assert window.collect_settings_state().batch_size == 0


def test_spinner_never_offers_negative(window):
    assert window.batch_spin.minimum() == 0
    assert window.batch_spin.maximum() > 0
    assert window.batch_spin.toolTip() == "Batch size (0 = auto)."


def test_tooltip_localized_en_fr(window):
    entry = localization.TRANSLATIONS["batch_size_tooltip"]
    assert set(entry) == {"en", "fr"}
    assert entry["en"] == "Batch size (0 = auto)."
    assert "0 = auto" in entry["fr"]

    window.language_combo.setCurrentText("Français")
    assert window.batch_spin.toolTip() == "Taille du lot (0 = auto)."
    window.language_combo.setCurrentText("English")
    assert window.batch_spin.toolTip() == "Batch size (0 = auto)."


# --------------------------------------------------------------------------
# Spinner navigation is continuous; passing through 1 never traps
# --------------------------------------------------------------------------
def test_spinner_0_to_1_to_2_to_3_succeeds(window):
    window.batch_spin.setValue(1)
    assert window.batch_spin.value() == 1
    assert window.batch_spin.isEnabled()
    assert window.boring_check.isChecked()  # 1 == Boring

    window.batch_spin.setValue(2)
    assert window.batch_spin.value() == 2
    assert window.batch_spin.isEnabled()
    assert not window.boring_check.isChecked()  # 2 -> Boring unchecked

    window.batch_spin.setValue(3)
    assert window.batch_spin.value() == 3
    assert window.batch_spin.isEnabled()
    assert not window.boring_check.isChecked()


def test_spinner_3_to_2_to_1_to_0_succeeds(window):
    window.batch_spin.setValue(3)
    window.batch_spin.setValue(2)
    assert window.batch_spin.isEnabled()
    assert not window.boring_check.isChecked()

    window.batch_spin.setValue(1)
    assert window.batch_spin.value() == 1
    assert window.batch_spin.isEnabled()
    assert window.boring_check.isChecked()

    window.batch_spin.setValue(0)
    assert window.batch_spin.value() == 0
    assert window.batch_spin.isEnabled()
    assert not window.boring_check.isChecked()
    assert window.collect_settings_state().batch_size == 0  # back to Auto


def test_at_one_boring_checked_and_spinner_enabled(window):
    window.batch_spin.setValue(1)
    assert window.boring_check.isChecked()
    # Phase B2: the spinner is NOT disabled at Boring.
    assert window.batch_spin.isEnabled()


def test_one_to_two_unchecks_boring(window):
    window.batch_spin.setValue(1)
    assert window.boring_check.isChecked()
    window.batch_spin.setValue(2)
    assert not window.boring_check.isChecked()
    assert window.batch_spin.value() == 2
    assert window.batch_spin.isEnabled()


def test_one_to_zero_unchecks_boring(window):
    window.batch_spin.setValue(1)
    assert window.boring_check.isChecked()
    window.batch_spin.setValue(0)
    assert not window.boring_check.isChecked()
    assert window.batch_spin.value() == 0
    assert window.batch_spin.isEnabled()


def test_checking_boring_from_five_pins_spinner_to_one_enabled(window):
    window.batch_spin.setValue(5)
    assert not window.boring_check.isChecked()
    window.boring_check.setChecked(True)
    assert window.batch_spin.value() == 1
    assert window.boring_check.isChecked()
    assert window.batch_spin.isEnabled()  # stays navigable


def test_unchecking_boring_at_one_resets_spinner_to_auto(window):
    window.boring_check.setChecked(True)
    assert window.batch_spin.value() == 1
    window.boring_check.setChecked(False)
    assert window.batch_spin.value() == 0
    assert not window.boring_check.isChecked()
    assert window.batch_spin.isEnabled()


# --------------------------------------------------------------------------
# Persisted load: legacy -1 -> Auto 0; 0 stays Auto 0
# --------------------------------------------------------------------------
def test_persisted_minus_one_loads_as_auto_zero(window):
    loaded = QtSettingsState.from_dict({"batch_size": -1})
    assert loaded.batch_size == 0  # normalized at the load boundary

    window._apply_state_to_controls(QtSettingsState.from_dict({"batch_size": -1}))
    assert window.batch_spin.value() == 0
    assert not window.boring_check.isChecked()
    assert window.collect_settings_state().batch_size == 0
    # The loaded model itself is canonical.
    assert window.settings_state.batch_size == 0


def test_persisted_zero_remains_auto_zero(window):
    loaded = QtSettingsState.from_dict({"batch_size": 0})
    assert loaded.batch_size == 0

    window._apply_state_to_controls(QtSettingsState.from_dict({"batch_size": 0}))
    assert window.batch_spin.value() == 0
    assert not window.boring_check.isChecked()
    assert window.collect_settings_state().batch_size == 0


def test_persisted_boring_one_still_loads_as_boring(window):
    loaded = QtSettingsState.from_dict({"batch_size": 1})
    assert loaded.batch_size == 1
    window._apply_state_to_controls(QtSettingsState.from_dict({"batch_size": 1}))
    assert window.batch_spin.value() == 1
    assert window.boring_check.isChecked()
    assert window.batch_spin.isEnabled()


# --------------------------------------------------------------------------
# Non-destructive Boring gating (§15): Drizzle request survives
# --------------------------------------------------------------------------
def test_drizzle_request_survives_0_1_2_pass_through(window):
    window.drizzle_check.setChecked(True)
    assert window.drizzle_check.isChecked()

    window.batch_spin.setValue(1)  # Boring: drizzle gated (visually cleared)
    assert window.boring_check.isChecked()
    assert window.drizzle_check.isChecked() is False  # gated, not erased
    assert window.drizzle_check.isEnabled() is False

    window.batch_spin.setValue(2)  # leave Boring
    assert not window.boring_check.isChecked()
    # Request restored: Drizzle is ON again at 2 (non-Boring).
    assert window.drizzle_check.isChecked() is True
    assert window.drizzle_check.isEnabled() is True
    assert window.collect_settings_state().use_drizzle is True


def test_no_drizzle_request_stays_off_through_pass_through(window):
    assert window.drizzle_check.isChecked() is False
    window.batch_spin.setValue(0)
    window.batch_spin.setValue(1)
    window.batch_spin.setValue(2)
    assert window.drizzle_check.isChecked() is False
    assert window.drizzle_check.isEnabled() is True
    assert window.collect_settings_state().use_drizzle is False


def test_drizzle_suboption_values_survive_boring_pass_through(window):
    # Every drizzle sub-option value must survive a 0 -> 1 -> 2 pass-through:
    # only enablement is gated under Boring, values are never overwritten.
    window.drizzle_check.setChecked(True)
    window.drizzle_mode_combo.setCurrentText("Large dataset")
    window.drizzle_group_spin.setValue(77)
    window.drizzle_scale_spin.setValue(3)
    window.drizzle_wht_spin.setValue(0.42)
    window.drizzle_kernel_combo.setCurrentText("square")
    window.drizzle_pixfrac_spin.setValue(0.88)

    window.batch_spin.setValue(1)
    assert window.drizzle_mode_combo.currentData() == "Incremental"
    window.batch_spin.setValue(2)

    state = window.collect_settings_state()
    assert state.use_drizzle is True
    assert state.drizzle_mode == "Incremental"
    assert state.drizzle_group_size == 77
    assert state.drizzle_scale == 3
    assert abs(state.drizzle_wht_threshold - 0.42) < 1e-9
    assert state.drizzle_kernel == "square"
    assert abs(state.drizzle_pixfrac - 0.88) < 1e-9


def test_boring_gate_restores_after_direct_check_uncheck(window):
    window.drizzle_check.setChecked(True)
    window.boring_check.setChecked(True)
    assert window.drizzle_check.isChecked() is False  # gated while active
    window.boring_check.setChecked(False)
    # Leaving Boring restores the requested Drizzle ON.
    assert window.drizzle_check.isChecked() is True
    assert window.drizzle_check.isEnabled() is True


def test_transient_boring_pass_has_no_model_side_effect(window):
    window.drizzle_check.setChecked(True)
    before = window.collect_settings_state()
    window.batch_spin.setValue(1)
    window.batch_spin.setValue(0)
    after = window.collect_settings_state()
    assert before.use_drizzle is True
    assert after.use_drizzle is True
    assert after.batch_size == 0  # back to Auto, request untouched


# --------------------------------------------------------------------------
# Launch routing (§16): effective Boring only
# --------------------------------------------------------------------------
def test_transient_crossing_one_does_not_route_boring(qapp):
    """A spinner pass 0 -> 1 -> 2 without Start never launches the boring
    subprocess route; Start at 2 goes through the normal controller path."""
    boring_factory_calls = []

    def factory():
        boring_factory_calls.append(True)
        raise AssertionError("boring runner must never be created here")

    win = MainWindow(boring_runner_factory=factory)
    controller_calls = []

    def spy_start(request, **kwargs):
        controller_calls.append(request)

    win.controller.start = spy_start
    try:
        win.batch_spin.setValue(0)
        win.batch_spin.setValue(1)  # transient Boring crossing…
        assert win.boring_check.isChecked()
        win.batch_spin.setValue(2)  # …left before Start
        assert not win.boring_check.isChecked()

        win.start_button.click()
        assert boring_factory_calls == []  # no boring runner was requested
        assert len(controller_calls) == 1  # normal controller route instead
    finally:
        win.shutdown()


def test_effective_boring_one_routes_to_boring_runner(qapp, tmp_path):
    """Start while the effective batch mode IS Boring (spinner at 1) must take
    the boring subprocess route — even though the spinner is enabled."""
    from seestar.gui_qt.boring_runner import BoringRunnerBase
    from seestar.gui_qt.boring_route import BoringRunRequest

    class FakeBoringRunner(BoringRunnerBase):
        def __init__(self):
            super().__init__()
            self.start_calls = []

        def start(self, request):
            self.start_calls.append(request)

        def cancel(self):
            pass

        def is_running(self):
            return bool(self.start_calls)

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "a.fits").write_bytes(b"")
    with open(input_dir / "stack_plan.csv", "w", encoding="utf-8") as handle:
        handle.write("a.fits\n")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    fakes = []

    def factory():
        fake = FakeBoringRunner()
        fakes.append(fake)
        return fake

    win = MainWindow(boring_runner_factory=factory)
    controller_calls = []

    def spy_start(request, **kwargs):
        controller_calls.append(request)

    win.controller.start = spy_start
    try:
        win.input_edit.setText(str(input_dir))
        win.output_edit.setText(str(output_dir))
        win.batch_spin.setValue(1)
        assert win.batch_spin.isEnabled()  # never disabled at Boring
        win.start_button.click()

        assert controller_calls == []  # NOT the normal controller route
        assert len(fakes) == 1 and len(fakes[0].start_calls) == 1
        assert isinstance(fakes[0].start_calls[0], BoringRunRequest)
        assert fakes[0].start_calls[0].batch_size == 1
    finally:
        win.shutdown()
