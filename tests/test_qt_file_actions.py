"""Offscreen Qt tests for path/file actions and reference/last-stack parity.

Covers the first block of Tk → Qt user-parity for file/folder paths and
actions, without touching the scientific backend:

* browse input / output / temp / reference / last-stack (monkeypatched dialogs),
* reference + last-stack controls surfaced on the Stacking tab and mirrored
  into ``QtSettingsState``,
* View Inputs (non-backend Qt dialog, main + staged folders),
* Open Output (``QDesktopServices.openUrl``, monkeypatched),
* Add Folder staging + validation (input/output/subfolder-of-output rejection),
* button enablement on path / run-state changes,
* ``_on_start`` passing staged additional folders into the ``RunRequest``.

``QT_QPA_PLATFORM=offscreen`` is set defensively before any ``QApplication`` is
created, mirroring the other Qt shell tests.  No real FITS parsing and no real
engine is involved.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QFileDialog

from seestar.gui_qt import MainWindow, create_application


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


def _dir_dialog(value):
    return lambda *a, **k: value


def _file_dialog(value):
    return lambda *a, **k: (value, "FITS files (*.fit *.fits)")


# --------------------------------------------------------------------------
# Browse path writes (monkeypatched file dialogs)
# --------------------------------------------------------------------------
def test_browse_input_writes_absolute_path(window, monkeypatch):
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(_dir_dialog("/inputs/raw"))
    )
    window._browse_input()
    assert window.input_edit.text() == os.path.abspath("/inputs/raw")
    assert window.collect_settings_state().input_folder == os.path.abspath("/inputs/raw")


def test_browse_output_writes_absolute_path(window, monkeypatch):
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(_dir_dialog("/outputs"))
    )
    window._browse_output()
    assert window.output_edit.text() == os.path.abspath("/outputs")
    assert window.collect_settings_state().output_folder == os.path.abspath("/outputs")


def test_browse_temp_writes_absolute_path(window, monkeypatch):
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(_dir_dialog("/tmp/stack"))
    )
    window._browse_temp()
    assert window.temp_edit.text() == os.path.abspath("/tmp/stack")
    assert window.collect_settings_state().temp_folder == os.path.abspath("/tmp/stack")


def test_browse_reference_writes_absolute_path(window, monkeypatch):
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(_file_dialog("/inputs/ref.fit"))
    )
    window._browse_reference()
    assert window.reference_edit.text() == os.path.abspath("/inputs/ref.fit")
    assert window._reference_origin_hint == "USER"
    assert (
        window.collect_settings_state().reference_image_path
        == os.path.abspath("/inputs/ref.fit")
    )


def test_browse_last_stack_writes_path_and_prefills_output(window, monkeypatch):
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(_file_dialog("/outputs/last.fit"))
    )
    window._browse_last_stack()
    assert window.last_stack_edit.text() == os.path.abspath("/outputs/last.fit")
    # Output was empty -> pre-filled from the selected file's parent (Tk parity).
    assert window.output_edit.text() == os.path.abspath("/outputs")
    assert window.collect_settings_state().last_stack_path == os.path.abspath("/outputs/last.fit")


def test_browse_last_stack_keeps_existing_output(window, monkeypatch):
    window.output_edit.setText("/keep/output")
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(_file_dialog("/outputs/last.fit"))
    )
    window._browse_last_stack()
    assert window.last_stack_edit.text() == os.path.abspath("/outputs/last.fit")
    assert window.output_edit.text() == "/keep/output"


def test_browse_cancelled_leaves_state_unchanged(window, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(_dir_dialog("")))
    window._browse_input()
    assert window.input_edit.text() == ""


# --------------------------------------------------------------------------
# Reference + last-stack controls surfaced on the Stacking tab
# --------------------------------------------------------------------------
def test_reference_and_last_stack_controls_exist(window):
    for attr in (
        "reference_edit",
        "last_stack_edit",
        "browse_reference_button",
        "browse_last_stack_button",
    ):
        assert hasattr(window, attr), f"missing path control {attr}"
    assert not window.reference_edit.isHidden()
    assert not window.last_stack_edit.isHidden()


def test_reference_and_last_stack_sync_to_state(window):
    window.reference_edit.setText("/inputs/ref.fit")
    window.last_stack_edit.setText("/outputs/last.fit")
    state = window.collect_settings_state()
    assert state.reference_image_path == "/inputs/ref.fit"
    assert state.last_stack_path == "/outputs/last.fit"


# --------------------------------------------------------------------------
# View Inputs (non-backend dialog, main + staged folders)
# --------------------------------------------------------------------------
def test_input_folder_summary_text(window, tmp_path):
    extra = tmp_path / "extra-a"
    extra.mkdir()
    window.input_edit.setText(str(tmp_path))
    window._additional_folders.append(str(extra))

    text = window._input_folder_summary_text()
    assert os.path.abspath(str(tmp_path)) in text
    assert os.path.abspath(str(extra)) in text


def test_input_folder_summary_empty_when_input_invalid(window):
    window.input_edit.setText("/does/not/exist")
    assert window._input_folder_summary_text() == ""


def test_view_inputs_action_opens_dialog(window, tmp_path, monkeypatch):
    window.input_edit.setText(str(tmp_path))
    window._sync_state_from_controls()
    captured = {}

    class FakeDialog:
        def exec(self):
            captured["exec"] = True
            return 0

    monkeypatch.setattr(
        window, "_build_input_folder_dialog", lambda: FakeDialog()
    )
    window._show_input_folder_list()
    assert captured.get("exec") is True


def test_view_inputs_invalid_logs_message(window, monkeypatch):
    window.input_edit.setText("/does/not/exist")
    window._sync_state_from_controls()
    logs = []
    monkeypatch.setattr(window, "log", lambda msg: logs.append(msg))
    window._show_input_folder_list()
    assert any("No valid input folder" in m for m in logs)


# --------------------------------------------------------------------------
# Open Output (monkeypatched desktop service)
# --------------------------------------------------------------------------
def test_open_output_action(window, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        staticmethod(lambda url: opened.append(url.toString()) or True),
    )
    window.output_edit.setText(str(tmp_path))
    window._open_output_folder()
    assert opened == [QUrl.fromLocalFile(str(tmp_path)).toString()]


def test_open_output_empty_no_crash(window):
    window.output_edit.setText("")
    window._open_output_folder()  # must not raise


def test_open_output_missing_no_crash(window, monkeypatch):
    opened = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        staticmethod(lambda url: opened.append(url.toString()) or True),
    )
    window.output_edit.setText("/does/not/exist/output")
    window._open_output_folder()  # must not raise
    assert opened == []


# --------------------------------------------------------------------------
# Add Folder staging / validation
# --------------------------------------------------------------------------
def test_add_folder_stages(window, tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    window.input_edit.setText(str(tmp_path))
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(_dir_dialog(str(extra)))
    )
    window._add_folder()
    assert os.path.abspath(str(extra)) in window._additional_folders


def test_add_folder_duplicate_not_double_staged(window, tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    window.input_edit.setText(str(tmp_path))
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(_dir_dialog(str(extra)))
    )
    window._add_folder()
    window._add_folder()
    assert window._additional_folders.count(os.path.abspath(str(extra))) == 1


def test_add_folder_rejects_input_folder(window, tmp_path):
    window.input_edit.setText(str(tmp_path))
    error = window._validate_additional_folder(str(tmp_path))
    assert error is not None
    assert "input" in error.lower()


def test_add_folder_rejects_output_folder(window, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    window.output_edit.setText(str(output))
    error = window._validate_additional_folder(str(output))
    assert error is not None
    assert "output" in error.lower()


def test_add_folder_rejects_output_subfolder(window, tmp_path):
    output = tmp_path / "output"
    sub = output / "sub"
    sub.mkdir(parents=True)
    window.output_edit.setText(str(output))
    error = window._validate_additional_folder(str(sub))
    assert error is not None
    assert "subfolder" in error.lower()


def test_add_folder_rejects_missing_folder(window):
    error = window._validate_additional_folder("/does/not/exist")
    assert error is not None
    assert "not found" in error.lower()


def test_add_folder_accepts_sibling_of_output(window, tmp_path):
    output = tmp_path / "output"
    sibling = tmp_path / "sibling"
    output.mkdir()
    sibling.mkdir()
    window.output_edit.setText(str(output))
    assert window._validate_additional_folder(str(sibling)) is None


def test_add_folder_while_running_logs_not_implemented(window, tmp_path):
    window.input_edit.setText(str(tmp_path))
    window._running = True
    window._additional_folders = []
    logs = []
    window.log = lambda msg: logs.append(msg)
    window._add_folder()
    assert window._additional_folders == []
    assert any("live add" in m for m in logs)


# --------------------------------------------------------------------------
# Button enablement
# --------------------------------------------------------------------------
def test_button_enablement_updates_on_path_changes(window, tmp_path):
    # Empty paths -> all disabled.
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

    window.input_edit.setText("")
    window._sync_state_from_controls()
    assert not window.view_inputs_button.isEnabled()
    assert not window.add_folder_button.isEnabled()


def test_add_folder_disabled_while_running(window, tmp_path):
    window.input_edit.setText(str(tmp_path))
    window._sync_state_from_controls()
    assert window.add_folder_button.isEnabled()

    window._running = True
    window._update_run_state()
    assert not window.add_folder_button.isEnabled()
    assert window.view_inputs_button.isEnabled()  # still viewable while running


# --------------------------------------------------------------------------
# _on_start passes staged additional folders into the request
# --------------------------------------------------------------------------
def test_on_start_passes_staged_folders(window, tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    window.input_edit.setText(str(tmp_path))
    window._additional_folders = [str(extra)]

    captured = {}
    monkeypatch.setattr(
        window.controller, "start", lambda request, **kw: captured.__setitem__("request", request)
    )
    window._on_start()
    req = captured["request"]
    assert req.backend_kwargs["initial_additional_folders"] == [str(extra)]
    # The snapshot must be a copy, not the live GUI list.
    assert req.backend_kwargs["initial_additional_folders"] is not window._additional_folders
