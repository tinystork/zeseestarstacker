"""Offscreen Qt tests for the ZeAnalyser launch seam (M7).

Covers, without touching the scientific backend and without ever spawning a
real ZeAnalyser process:

* Analyse button enablement (disabled initially / for invalid input, enabled
  for an existing input directory, updated on input-path changes),
* ``MainWindow._on_analyse`` success / missing-analyzer / launch-exception
  paths via the injectable launcher + command-file maker,
* the single-shot command-file reference consumption seam,
* the stdlib-only ``seestar.gui_qt.analyzer_launch`` helpers (exact
  command/env, detection fallback, temp command-file path, reference parsing),
* source hygiene for the new seam module.

``QT_QPA_PLATFORM=offscreen`` is set defensively before any ``QApplication``
is created, mirroring the other Qt shell tests.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt import analyzer_launch


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


# --------------------------------------------------------------------------
# Analyse button enablement
# --------------------------------------------------------------------------
def test_analyse_button_disabled_initially(window):
    assert not window.analyse_button.isEnabled()


def test_analyse_button_disabled_for_invalid_input(window):
    window.input_edit.setText("/does/not/exist")
    window._sync_state_from_controls()
    assert not window.analyse_button.isEnabled()


def test_analyse_button_disabled_for_empty_input(window):
    window.input_edit.setText("")
    window._sync_state_from_controls()
    assert not window.analyse_button.isEnabled()


def test_analyse_button_enabled_for_existing_dir(window, tmp_path):
    window.input_edit.setText(str(tmp_path))
    window._sync_state_from_controls()
    assert window.analyse_button.isEnabled()


def test_analyse_button_updates_on_input_change(window, tmp_path):
    assert not window.analyse_button.isEnabled()
    window.input_edit.setText(str(tmp_path))
    window._sync_state_from_controls()
    assert window.analyse_button.isEnabled()
    window.input_edit.setText("/does/not/exist")
    window._sync_state_from_controls()
    assert not window.analyse_button.isEnabled()
    window.input_edit.setText("")
    window._sync_state_from_controls()
    assert not window.analyse_button.isEnabled()


def test_current_language_code(window):
    assert window._current_language_code() == "en"
    window.language_combo.setCurrentText("Français")
    assert window._current_language_code() == "fr"


# --------------------------------------------------------------------------
# _on_analyse launch paths (injectable launcher + maker; no real spawn)
# --------------------------------------------------------------------------
def test_analyse_success_builds_logs_and_does_not_run(window, tmp_path, monkeypatch):
    window.input_edit.setText(str(tmp_path))
    window._sync_state_from_controls()

    calls = {}
    monkeypatch.setattr(
        window, "_analyzer_command_file_maker", lambda: "/tmp/analyzer_cmd.txt"
    )

    def fake_launcher(input_folder, lang, command_file_path):
        calls["input_folder"] = input_folder
        calls["lang"] = lang
        calls["command_file_path"] = command_file_path
        return True

    monkeypatch.setattr(window, "_analyzer_launcher", fake_launcher)

    window._on_analyse()

    assert calls["input_folder"] == str(tmp_path)
    assert calls["lang"] == "en"
    assert calls["command_file_path"] == "/tmp/analyzer_cmd.txt"
    assert window._analyzer_command_file_path == "/tmp/analyzer_cmd.txt"
    assert "Analyzer launched" in window.log_view.toPlainText()
    assert "Analyzer launched" in window.statusBar().currentMessage()
    # Launching ZeAnalyser must never mark a run active.
    assert window.is_running is False
    assert window.start_button.isEnabled()


def test_analyse_missing_analyzer_logs_failure_no_crash(window, tmp_path, monkeypatch):
    window.input_edit.setText(str(tmp_path))
    window._sync_state_from_controls()
    monkeypatch.setattr(
        window, "_analyzer_command_file_maker", lambda: "/tmp/analyzer_cmd.txt"
    )
    monkeypatch.setattr(window, "_analyzer_launcher", lambda i, l, c: False)

    window._on_analyse()  # must not raise

    assert "ZeAnalyser not found" in window.log_view.toPlainText()
    assert "ZeAnalyser not found" in window.statusBar().currentMessage()
    assert window.is_running is False


def test_analyse_launch_exception_logs_failure_no_crash(window, tmp_path, monkeypatch):
    window.input_edit.setText(str(tmp_path))
    window._sync_state_from_controls()
    monkeypatch.setattr(
        window, "_analyzer_command_file_maker", lambda: "/tmp/analyzer_cmd.txt"
    )

    def boom(i, l, c):
        raise OSError("spawn failed")

    monkeypatch.setattr(window, "_analyzer_launcher", boom)

    window._on_analyse()  # must not raise

    assert "launch failed" in window.log_view.toPlainText()
    assert "launch failed" in window.statusBar().currentMessage()
    assert window.is_running is False


def test_analyse_command_file_creation_error_no_crash(window, tmp_path, monkeypatch):
    window.input_edit.setText(str(tmp_path))
    window._sync_state_from_controls()
    launched = []
    monkeypatch.setattr(window, "_analyzer_launcher", lambda i, l, c: launched.append(c) or True)

    def bad_maker():
        raise OSError("cannot create temp dir")

    monkeypatch.setattr(window, "_analyzer_command_file_maker", bad_maker)

    window._on_analyse()  # must not raise

    assert launched == []  # launcher never invoked
    assert "cannot create command file" in window.log_view.toPlainText()
    assert window.is_running is False


def test_analyse_invalid_input_no_launch(window, monkeypatch):
    launched = []
    monkeypatch.setattr(window, "_analyzer_launcher", lambda i, l, c: launched.append((i, l, c)) or True)
    window.input_edit.setText("/does/not/exist")
    window._sync_state_from_controls()

    window._on_analyse()

    assert launched == []
    assert "select a valid input folder" in window.log_view.toPlainText()


# --------------------------------------------------------------------------
# Single-shot command-file reference consumption seam
# --------------------------------------------------------------------------
def test_check_analyzer_command_file_no_path(window):
    assert window._analyzer_command_file_path is None
    assert window._check_analyzer_command_file() is None


def test_check_analyzer_command_file_updates_reference(tmp_path):
    ref_file = tmp_path / "ref.fit"
    ref_file.write_text("", encoding="utf-8")
    cmd_file = tmp_path / "analyzer_stack_command_1.txt"
    cmd_file.write_text(f"REFERENCE={ref_file}\nTIMESTAMP=2026-08-19T21:00:00\n", encoding="utf-8")

    win = MainWindow()
    try:
        win._analyzer_command_file_path = str(cmd_file)
        ref = win._check_analyzer_command_file()
        assert ref == str(ref_file)
        assert win.reference_edit.text() == str(ref_file)
        assert win.collect_settings_state().reference_image_path == str(ref_file)
        assert not cmd_file.exists()  # consumed best-effort delete
    finally:
        win.shutdown()


def test_check_analyzer_command_file_empty_reference_no_update(tmp_path):
    cmd_file = tmp_path / "analyzer_stack_command_2.txt"
    cmd_file.write_text("TIMESTAMP=2026-08-19T21:00:00\n", encoding="utf-8")

    win = MainWindow()
    try:
        win.reference_edit.setText("/existing/ref.fit")
        win._analyzer_command_file_path = str(cmd_file)
        ref = win._check_analyzer_command_file()
        assert ref is None
        assert win.reference_edit.text() == "/existing/ref.fit"
        assert not cmd_file.exists()
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# stdlib-only seam helpers (exact command/env, detection, temp path, parsing)
# --------------------------------------------------------------------------
def test_detect_analyzer_command_prefers_entry_point():
    def which(name):
        return "/usr/bin/zeanalyser" if name == "zeanalyser" else None

    def find_spec(name):
        return None

    assert analyzer_launch.detect_analyzer_command(which=which, find_spec=find_spec) == [
        "/usr/bin/zeanalyser"
    ]


def test_detect_analyzer_command_falls_back_to_module(monkeypatch):
    import sys

    def which(name):
        return None

    def find_spec(name):
        return object() if name == "zeanalyser" else None

    assert analyzer_launch.detect_analyzer_command(which=which, find_spec=find_spec) == [
        sys.executable,
        "-m",
        "zeanalyser",
    ]


def test_detect_analyzer_command_absent():
    assert analyzer_launch.detect_analyzer_command(which=lambda n: None, find_spec=lambda n: None) is None


def test_build_analyzer_command_exact():
    cmd = analyzer_launch.build_analyzer_command(
        "/data/lights", "en", detect=lambda: ["/usr/bin/zeanalyser"]
    )
    assert cmd == [
        "/usr/bin/zeanalyser",
        "--input-dir",
        "/data/lights",
        "--lang",
        "en",
        "--lock-lang",
    ]


def test_build_analyzer_command_none_when_absent():
    assert analyzer_launch.build_analyzer_command("/data/lights", "en", detect=lambda: None) is None


def test_launch_analyzer_spawns_with_env():
    calls = {}

    def fake_popen(cmd, env=None):
        calls["cmd"] = cmd
        calls["env"] = env

    result = analyzer_launch.launch_analyzer(
        "/data/lights",
        "fr",
        "/tmp/cmd.txt",
        popen=fake_popen,
        detect=lambda: ["/usr/bin/zeanalyser"],
    )
    assert result is True
    assert calls["cmd"] == [
        "/usr/bin/zeanalyser",
        "--input-dir",
        "/data/lights",
        "--lang",
        "fr",
        "--lock-lang",
    ]
    assert calls["env"]["ZEANALYSER_COMMAND_FILE"] == "/tmp/cmd.txt"


def test_launch_analyzer_no_popen_when_absent():
    called = []

    def fake_popen(cmd, env=None):
        called.append(cmd)

    result = analyzer_launch.launch_analyzer(
        "/data/lights", "en", "/tmp/cmd.txt", popen=fake_popen, detect=lambda: None
    )
    assert result is False
    assert called == []


def test_make_command_file_path(tmp_path):
    path = analyzer_launch.make_command_file_path(
        gettempdir=lambda: str(tmp_path), getpid=lambda: 4242
    )
    assert path == str(tmp_path / "seestar_stacker_comm" / "analyzer_stack_command_4242.txt")
    assert (tmp_path / "seestar_stacker_comm").is_dir()
    # The directory exists but the file itself is created by ZeAnalyser.
    assert not os.path.exists(path)


def test_parse_reference_and_consume(tmp_path):
    content = "REFERENCE=/data/ref.fit\nTIMESTAMP=2026-08-19T21:00:00\nSOME_OTHER=foo\n"
    assert analyzer_launch.parse_reference_from_command_file(content) == "/data/ref.fit"
    assert analyzer_launch.parse_reference_from_command_file("") is None
    assert analyzer_launch.parse_reference_from_command_file("TIMESTAMP=x\n") is None

    cmd_file = tmp_path / "cmd.txt"
    cmd_file.write_text(content, encoding="utf-8")
    ref = analyzer_launch.consume_command_file(str(cmd_file))
    assert ref == "/data/ref.fit"
    assert not cmd_file.exists()


# --------------------------------------------------------------------------
# Source / import hygiene for the new seam
# --------------------------------------------------------------------------
def test_analyzer_launch_source_is_stdlib_only():
    from pathlib import Path

    path = Path(analyzer_launch.__file__).resolve()
    text = path.read_text(encoding="utf-8")
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
        "import numpy",
        "import cv2",
    )
    for token in forbidden:
        assert token not in text, f"analyzer_launch.py references {token}"


def test_analyzer_launch_importable_without_qt():
    """The seam is stdlib-only: loadable by file path without PySide6 or Tk.

    Loading through ``seestar.gui_qt`` would pull in the whole Qt shell
    package, so the file is loaded directly (mirroring
    ``tests/test_analyzer_launch.py``) to prove the seam itself imports
    neither PySide6 nor Tk nor the engine.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    seam_path = root / "seestar" / "gui_qt" / "analyzer_launch.py"
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('_seam', {str(seam_path)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "assert m.COMMAND_FILE_ENV_VAR == 'ZEANALYSER_COMMAND_FILE'\n"
        "bad = [x for x in sys.modules if x.startswith('PySide6') or x.startswith('tkinter')]\n"
        "assert not bad, bad\n"
        "print('SEAM_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert proc.returncode == 0, (
        f"seam import failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
