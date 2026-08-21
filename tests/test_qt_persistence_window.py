"""M8 seam tests: Qt settings persistence wiring in ``MainWindow``.

Offscreen tests for the bounded, reversible settings/geometry persistence:

* ``MainWindow`` can be constructed with an injected ``settings_path`` so tests
  never touch a real home/CWD file,
* on construction, persisted settings are applied to the visible controls,
* on shutdown (or an explicit save), ``collect_settings_state()`` is written to
  the injected JSON (paths, last-stack path, filename, batch/stacking/
  final-combine/drizzle/solver fields),
* window geometry is written as a base64 key and re-applied without crash,
* path-action enablement is refreshed for existing vs non-existing dirs,
* a missing/corrupt file degrades to defaults without crashing.

``QT_QPA_PLATFORM=offscreen`` is set defensively before any ``QApplication`` is
created.  No real stacking, no engine, no Tk.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.settings_state import QtSettingsState


@pytest.fixture(scope="session", autouse=True)
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _settings_path(tmp_path):
    return str(tmp_path / "seestar_settings.json")


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return path


# --------------------------------------------------------------------------
# Construction loads + applies injected settings
# --------------------------------------------------------------------------
def test_construct_loads_injected_settings_into_controls(tmp_path):
    p = _settings_path(tmp_path)
    _write_json(
        p,
        {
            "input_folder": "/inputs/raw",
            "output_folder": "/outputs",
            "temp_folder": "/tmp/stack",
            "output_filename": "stack.fits",
            "reference_image_path": "/inputs/ref.fit",
            "last_stack_path": "/outputs/last.fit",
            "batch_size": 4,
            "stacking_mode": "median",
            "stack_final_combine": "reproject",
            "use_drizzle": True,
            "drizzle_mode": "Incremental",
            "drizzle_group_size": 77,
            "local_solver_preference": "astap",
            "kappa": 3.5,
        },
    )

    win = MainWindow(settings_path=p)
    try:
        assert win.input_edit.text() == "/inputs/raw"
        assert win.output_edit.text() == "/outputs"
        assert win.temp_edit.text() == "/tmp/stack"
        assert win.output_filename_edit.text() == "stack.fits"
        assert win.reference_edit.text() == "/inputs/ref.fit"
        assert win.last_stack_edit.text() == "/outputs/last.fit"
        assert win.batch_spin.value() == 4
        assert win.stacking_mode_combo.currentText() == "median"
        assert win.drizzle_check.isChecked() is True
        assert win.drizzle_mode_combo.currentText() == "Large dataset"
        assert win.drizzle_group_spin.value() == 77
        assert win.solver_combo.currentText() == "astap"
        assert win._settings_widgets["kappa"].value() == pytest.approx(3.5)

        state = win.collect_settings_state()
        assert state.stack_final_combine == "reproject"
        assert state.reproject_between_batches is True
        assert state.reproject_coadd_final is False
    finally:
        win.shutdown()


def test_construct_missing_settings_file_uses_defaults(tmp_path):
    win = MainWindow(settings_path=_settings_path(tmp_path))
    try:
        state = win.collect_settings_state()
        assert state.to_dict() == QtSettingsState.defaults()
    finally:
        win.shutdown()


def test_construct_corrupt_settings_file_uses_defaults(tmp_path):
    p = _settings_path(tmp_path)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("{ this is not json")
    win = MainWindow(settings_path=p)
    try:
        assert win.collect_settings_state().to_dict() == QtSettingsState.defaults()
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Save on shutdown / explicit save
# --------------------------------------------------------------------------
def test_shutdown_saves_current_settings_to_injected_json(tmp_path):
    p = _settings_path(tmp_path)
    win = MainWindow(settings_path=p)
    win.input_edit.setText(str(tmp_path / "inputs"))
    win.output_edit.setText(str(tmp_path / "outputs"))
    win.last_stack_edit.setText(str(tmp_path / "outputs" / "last.fit"))
    win.batch_spin.setValue(4)
    win.solver_combo.setCurrentText("astap")
    win._settings_widgets["kappa"].setValue(3.5)
    win._settings_widgets["correct_hot_pixels"].setChecked(False)
    win.shutdown()

    assert os.path.exists(p)
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["input_folder"] == str(tmp_path / "inputs")
    assert data["output_folder"] == str(tmp_path / "outputs")
    assert data["last_stack_path"] == str(tmp_path / "outputs" / "last.fit")
    assert data["batch_size"] == 4
    assert data["local_solver_preference"] == "astap"
    assert data["kappa"] == pytest.approx(3.5)
    assert data["correct_hot_pixels"] is False
    assert "window_geometry" in data


def test_explicit_save_persists_settings(tmp_path):
    p = _settings_path(tmp_path)
    win = MainWindow(settings_path=p)
    win.output_edit.setText("/outputs")
    win.output_filename_edit.setText("stack.fits")
    win._save_persisted_settings()

    assert os.path.exists(p)
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["output_folder"] == "/outputs"
    assert data["output_filename"] == "stack.fits"
    assert "window_geometry" in data
    win.shutdown()


def test_no_settings_path_writes_nothing(tmp_path, monkeypatch):
    # Bare construction (persistence disabled) must not create a file anywhere.
    monkeypatch.chdir(tmp_path)
    win = MainWindow()
    win.shutdown()
    assert not os.path.exists(os.path.join(str(tmp_path), "seestar_settings.json"))


# --------------------------------------------------------------------------
# Geometry round trip
# --------------------------------------------------------------------------
def test_geometry_round_trip_without_crash(tmp_path):
    p = _settings_path(tmp_path)
    win = MainWindow(settings_path=p)
    win.resize(480, 320)
    key = win._geometry_to_key()
    assert isinstance(key, str)
    assert key  # non-empty offscreen geometry
    # Re-applying the same geometry must not crash.
    assert win._restore_geometry_from_key(key) in (True, False)
    win.shutdown()

    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Geometry is saved as a non-empty base64 string; the exact bytes may
    # legitimately differ between two ``saveGeometry()`` calls (offscreen size
    # clamping), so we only require a non-empty, re-appliable string.
    assert isinstance(data["window_geometry"], str)
    assert data["window_geometry"]

    # A fresh window re-applies the persisted geometry without crashing.
    win2 = MainWindow(settings_path=p)
    win2.shutdown()


def test_restore_geometry_ignores_garbage(tmp_path):
    win = MainWindow()
    assert win._restore_geometry_from_key("!!!not-base64!!!") in (True, False)
    win.shutdown()


# --------------------------------------------------------------------------
# Path action state refresh after load
# --------------------------------------------------------------------------
def test_path_actions_refreshed_for_existing_dirs(tmp_path):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()

    p = _settings_path(tmp_path)
    _write_json(p, {"input_folder": str(inputs), "output_folder": str(outputs)})

    win = MainWindow(settings_path=p)
    try:
        assert win.view_inputs_button.isEnabled()
        assert win.open_output_button.isEnabled()
        assert win.add_folder_button.isEnabled()
        assert win.analyse_button.isEnabled()
    finally:
        win.shutdown()


def test_path_actions_refreshed_for_missing_dirs(tmp_path):
    p = _settings_path(tmp_path)
    _write_json(
        p,
        {
            "input_folder": str(tmp_path / "does" / "not" / "exist"),
            "output_folder": str(tmp_path / "also" / "missing"),
        },
    )

    win = MainWindow(settings_path=p)
    try:
        assert not win.view_inputs_button.isEnabled()
        assert not win.open_output_button.isEnabled()
        assert not win.add_folder_button.isEnabled()
        assert not win.analyse_button.isEnabled()
    finally:
        win.shutdown()


def test_loaded_paths_persist_through_state(tmp_path):
    """last_stack_path and output/input/temp/reference survive a save→load."""
    p = _settings_path(tmp_path)
    win = MainWindow(settings_path=p)
    win.input_edit.setText("/in")
    win.output_edit.setText("/out")
    win.temp_edit.setText("/tmp")
    win.reference_edit.setText("/in/ref.fit")
    win.last_stack_edit.setText("/out/last.fit")
    win.shutdown()

    win2 = MainWindow(settings_path=p)
    try:
        st = win2.collect_settings_state()
        assert st.input_folder == "/in"
        assert st.output_folder == "/out"
        assert st.temp_folder == "/tmp"
        assert st.reference_image_path == "/in/ref.fit"
        assert st.last_stack_path == "/out/last.fit"
    finally:
        win2.shutdown()
