"""M6 seam tests: Qt-side preflight validation for real-backend starts.

These tests exercise :func:`seestar.gui_qt.settings_validation.
validate_settings_for_backend` — the pure, engine/Tk/Qt-free validation seam —
with **no QApplication and no engine**:

* ``simulated`` mode is always permissive (empty folders return no errors),
* ``seestar`` mode rejects empty input/output folders,
* ``batch_size`` must be integer-like and ``>= -1``,
* ``drizzle_group_size`` must be ``> 0``,
* the same validation works against both a :class:`QtSettingsState` and the
  canonical :class:`RunRequest` (reading ``input_dir``/``output_dir`` from its
  ``backend_kwargs``).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from seestar.gui_qt.settings_validation import validate_settings_for_backend
from seestar.gui_qt.run_bridge import build_run_request
from seestar.gui_qt.settings_state import QtSettingsState


def _state(**overrides) -> QtSettingsState:
    state = QtSettingsState()
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _request(**overrides) -> object:
    return build_run_request(_state(**overrides))


# --------------------------------------------------------------------------
# Mode policy
# --------------------------------------------------------------------------
def test_simulated_mode_is_permissive_with_empty_folders():
    state = _state()  # defaults: empty input/output
    assert validate_settings_for_backend(state, "simulated") == []


def test_unknown_mode_is_permissive():
    state = _state()
    assert validate_settings_for_backend(state, "bogus") == []


# --------------------------------------------------------------------------
# seestar mode: folders
# --------------------------------------------------------------------------
def test_seestar_mode_rejects_empty_folders():
    errors = validate_settings_for_backend(_state(), "seestar")
    assert "Input folder is empty." in errors
    assert "Output folder is empty." in errors


def test_seestar_mode_accepts_minimally_valid_settings():
    state = _state(input_folder="/in", output_folder="/out")
    assert validate_settings_for_backend(state, "seestar") == []


def test_seestar_mode_whitespace_folder_is_empty():
    errors = validate_settings_for_backend(
        _state(input_folder="   ", output_folder="\t"), "seestar"
    )
    assert "Input folder is empty." in errors
    assert "Output folder is empty." in errors


# --------------------------------------------------------------------------
# seestar mode: batch size
# --------------------------------------------------------------------------
def test_batch_size_must_be_integer_like():
    errors = validate_settings_for_backend(
        _state(input_folder="/in", output_folder="/out", batch_size="abc"),
        "seestar",
    )
    assert any("Batch size must be an integer" in e for e in errors)


def test_batch_size_below_auto_sentinel_rejected():
    errors = validate_settings_for_backend(
        _state(input_folder="/in", output_folder="/out", batch_size=-5),
        "seestar",
    )
    assert any("Batch size must be -1 (auto) or greater" in e for e in errors)


def test_batch_size_valid_sentinels_accepted():
    for batch in (-1, 0, 1, 2, 100):
        state = _state(input_folder="/in", output_folder="/out", batch_size=batch)
        assert validate_settings_for_backend(state, "seestar") == [], batch


# --------------------------------------------------------------------------
# seestar mode: drizzle group size
# --------------------------------------------------------------------------
def test_drizzle_group_size_must_be_positive():
    for bad in (0, -3):
        errors = validate_settings_for_backend(
            _state(
                input_folder="/in",
                output_folder="/out",
                drizzle_group_size=bad,
            ),
            "seestar",
        )
        assert any(
            "Drizzle group size must be greater than 0" in e for e in errors
        ), bad


def test_drizzle_group_size_must_be_integer_like():
    errors = validate_settings_for_backend(
        _state(
            input_folder="/in",
            output_folder="/out",
            drizzle_group_size="lots",
        ),
        "seestar",
    )
    assert any("Drizzle group size must be an integer" in e for e in errors)


def test_drizzle_group_size_valid_accepted():
    state = _state(
        input_folder="/in", output_folder="/out", drizzle_group_size=1
    )
    assert validate_settings_for_backend(state, "seestar") == []


# --------------------------------------------------------------------------
# RunRequest support (reads backend_kwargs key names)
# --------------------------------------------------------------------------
def test_run_request_with_empty_folders_rejected():
    errors = validate_settings_for_backend(_request(), "seestar")
    assert "Input folder is empty." in errors
    assert "Output folder is empty." in errors


def test_run_request_with_valid_settings_accepted():
    req = _request(input_folder="/in", output_folder="/out")
    assert validate_settings_for_backend(req, "seestar") == []


def test_run_request_simulated_mode_permissive():
    req = _request()  # empty folders
    assert validate_settings_for_backend(req, "simulated") == []


def test_run_request_carries_batch_and_drizzle_values():
    req = _request(
        input_folder="/in",
        output_folder="/out",
        batch_size="nope",
        drizzle_group_size=0,
    )
    errors = validate_settings_for_backend(req, "seestar")
    assert any("Batch size must be an integer" in e for e in errors)
    assert any("Drizzle group size must be greater than 0" in e for e in errors)
