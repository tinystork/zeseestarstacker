"""M25.5-F final parity audit seam tests.

These are the *documentation-truth* guards for the final audit (checklist
section 23): the three Expert-tab enabler flags ``apply_bn`` / ``apply_cb`` /
``apply_final_crop`` are **GUI gating controls only** — they gate their
sub-option widgets and are persisted/restored like the Tk
``apply_bn_var`` / ``apply_cb_var`` / ``apply_final_crop_var``, but the engine
does **not** consume them today.  This is a deliberate, Tk-identical non-blocker
(not a new backend semantic): the shared ``build_backend_kwargs`` omits the
three flags on *both* the Tk and the Qt side, so the Qt shell is at exact
parity with Tk.

These tests assert that documented truth without touching the engine, Tk, or
any scientific code: they read only the canonical stdlib builder
(``seestar.gui.run_config``, reached via ``seestar.gui_qt.run_bridge``) and the
pure-stdlib ``QtSettingsState`` / the enabler-gates table.

No FITS/PNG writes, no subprocess, no ``QApplication`` required.
"""

from __future__ import annotations

from seestar.gui_qt.main_window import EXPERT_ENABLER_GATES
from seestar.gui_qt.run_bridge import build_backend_kwargs
from seestar.gui_qt.settings_state import QtSettingsState

# The three Expert-tab enabler flags that must remain GUI-gating-only.
NON_BLOCKER_ENABLER_FLAGS = ("apply_bn", "apply_cb", "apply_final_crop")


def test_apply_enabler_flags_not_consumed_by_build_backend_kwargs():
    """The three enabler flags are NOT in the engine kwargs (Tk-identical).

    ``build_backend_kwargs`` is the *shared* Tk/Qt builder: the Tk GUI calls it
    too (via ``build_run_request``), so if the flags are absent here they are
    absent for Tk as well — the Qt shell is at exact parity, not a gap.
    """
    kwargs = build_backend_kwargs(QtSettingsState())
    for flag in NON_BLOCKER_ENABLER_FLAGS:
        assert flag not in kwargs, (
            f"{flag!r} leaked into build_backend_kwargs; it must stay a "
            "GUI-gating control, not a backend keyword"
        )


def test_apply_enabler_flags_are_persisted_gui_state():
    """The three flags exist in the model defaults (persisted GUI state).

    They are GUI-gating controls that round-trip through the settings surface
    like the Tk ``apply_bn_var`` / ``apply_cb_var`` / ``apply_final_crop_var``
    — but that persistence is GUI-state persistence, not engine consumption.
    """
    defaults = QtSettingsState.defaults()
    for flag in NON_BLOCKER_ENABLER_FLAGS:
        assert flag in defaults, f"missing persisted GUI flag {flag!r}"
        assert isinstance(defaults[flag], bool), flag


def test_apply_enabler_flags_are_gating_only():
    """Each of the three flags is a gating enabler with the Tk sub-options.

    The gate table drives ``_update_expert_enabler_states`` (the Qt equivalent
    of the Tk ``_update_bn_options_state`` / ``_update_cb_options_state`` /
    ``_update_crop_options_state``): unchecked disables the gated widgets,
    checked re-enables them.  This is the *only* runtime effect of the flags.
    """
    expected_gates = {
        "apply_bn": [
            "bn_grid_size_str",
            "bn_perc_low",
            "bn_perc_high",
            "bn_std_factor",
            "bn_min_gain",
            "bn_max_gain",
        ],
        "apply_cb": [
            "cb_border_size",
            "cb_blur_radius",
            "cb_min_b_factor",
            "cb_max_b_factor",
        ],
        "apply_final_crop": ["final_edge_crop_percent"],
    }
    for flag, gated in expected_gates.items():
        assert flag in EXPERT_ENABLER_GATES, f"missing enabler gate {flag!r}"
        assert list(EXPERT_ENABLER_GATES[flag]) == gated, flag
