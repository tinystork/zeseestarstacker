"""Tests for the ZeAnalyser launch helpers (``seestar/gui/analyzer_launch.py``).

The module is loaded directly by file path (mirroring the pattern used by
``tests/test_solver_config.py``) so these tests do not pull in the heavy
``seestar`` package tree, which requires optional dependencies (OpenCV,
Pillow.ImageTk) that are absent from this test environment.
"""

import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "seestar_analyzer_launch", ROOT / "seestar" / "gui" / "analyzer_launch.py"
)
analyzer_launch = importlib.util.module_from_spec(spec)
sys.modules["seestar_analyzer_launch"] = analyzer_launch
spec.loader.exec_module(analyzer_launch)


def _set_entry_point(monkeypatch, exe, product="zeanalyser"):
    monkeypatch.setattr(
        shutil, "which", lambda name: exe if name == product else None
    )


def _set_module_present(monkeypatch, present, product="zeanalyser"):
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if (present and name == product) else None,
    )


def test_build_command_prefers_entry_point(monkeypatch):
    _set_entry_point(monkeypatch, "/usr/bin/zeanalyser")
    _set_module_present(monkeypatch, False)
    cmd = analyzer_launch.build_analyzer_command("/data/lights", "en")
    assert cmd == [
        "/usr/bin/zeanalyser",
        "--input-dir",
        "/data/lights",
        "--lang",
        "en",
        "--lock-lang",
    ]


def test_build_command_falls_back_to_module(monkeypatch):
    _set_entry_point(monkeypatch, None)
    _set_module_present(monkeypatch, True)
    cmd = analyzer_launch.build_analyzer_command("/data/lights", "fr")
    assert cmd == [
        sys.executable,
        "-m",
        "zeanalyser",
        "--input-dir",
        "/data/lights",
        "--lang",
        "fr",
        "--lock-lang",
    ]


def test_build_command_returns_none_when_absent(monkeypatch):
    _set_entry_point(monkeypatch, None)
    _set_module_present(monkeypatch, False)
    assert analyzer_launch.build_analyzer_command("/data/lights", "en") is None


def test_make_analyzer_env_sets_command_file_and_preserves_existing():
    env = analyzer_launch.make_analyzer_env(
        "/tmp/cmd.txt", base_env={"PATH": "/bin", "FOO": "bar"}
    )
    assert env["ZEANALYSER_COMMAND_FILE"] == "/tmp/cmd.txt"
    assert env["PATH"] == "/bin"
    assert env["FOO"] == "bar"


def test_launch_analyzer_spawns_process_with_env(monkeypatch):
    _set_entry_point(monkeypatch, "/usr/bin/zeanalyser")
    _set_module_present(monkeypatch, False)

    calls = {}

    def fake_popen(cmd, env=None):
        calls["cmd"] = cmd
        calls["env"] = env
        return object()

    result = analyzer_launch.launch_analyzer(
        "/data/lights", "en", "/tmp/cmd.txt", popen=fake_popen
    )
    assert result is True
    assert calls["cmd"] == [
        "/usr/bin/zeanalyser",
        "--input-dir",
        "/data/lights",
        "--lang",
        "en",
        "--lock-lang",
    ]
    assert calls["env"]["ZEANALYSER_COMMAND_FILE"] == "/tmp/cmd.txt"


def test_launch_analyzer_no_popen_when_absent(monkeypatch):
    _set_entry_point(monkeypatch, None)
    _set_module_present(monkeypatch, False)

    called = []

    def fake_popen(cmd, env=None):
        called.append(cmd)

    result = analyzer_launch.launch_analyzer(
        "/data/lights", "en", "/tmp/cmd.txt", popen=fake_popen
    )
    assert result is False
    assert called == []


def test_parse_reference_extracts_reference_and_ignores_timestamp():
    content = "REFERENCE=/data/ref.fit\nTIMESTAMP=2026-08-19T21:00:00\nSOME_OTHER=foo\n"
    assert analyzer_launch.parse_reference_from_command_file(content) == "/data/ref.fit"


def test_parse_reference_returns_none_without_reference():
    assert analyzer_launch.parse_reference_from_command_file("TIMESTAMP=2026-08-19T21:00:00\n") is None
    assert analyzer_launch.parse_reference_from_command_file("") is None


def test_consume_command_file_reads_deletes_and_returns_reference(tmp_path):
    cmd_file = tmp_path / "analyzer_stack_command_1234.txt"
    cmd_file.write_text("REFERENCE=/data/ref.fit\nTIMESTAMP=2026-08-19T21:00:00\n", encoding="utf-8")

    ref = analyzer_launch.consume_command_file(str(cmd_file))

    assert ref == "/data/ref.fit"
    assert not cmd_file.exists()


def test_consume_command_file_without_reference_deletes_file(tmp_path):
    cmd_file = tmp_path / "analyzer_stack_command_1234.txt"
    cmd_file.write_text("TIMESTAMP=2026-08-19T21:00:00\n", encoding="utf-8")

    ref = analyzer_launch.consume_command_file(str(cmd_file))

    assert ref is None
    assert not cmd_file.exists()


def test_detect_zemosaic_prefers_entry_point(monkeypatch):
    _set_entry_point(monkeypatch, "/usr/bin/zemosaic", product="zemosaic")
    _set_module_present(monkeypatch, False, product="zemosaic")
    assert analyzer_launch.detect_zemosaic_command() == ["/usr/bin/zemosaic"]


def test_detect_zemosaic_falls_back_to_module(monkeypatch):
    _set_entry_point(monkeypatch, None, product="zemosaic")
    _set_module_present(monkeypatch, True, product="zemosaic")
    assert analyzer_launch.detect_zemosaic_command() == [
        sys.executable,
        "-m",
        "zemosaic",
    ]


def test_detect_zemosaic_returns_none_when_absent(monkeypatch):
    _set_entry_point(monkeypatch, None, product="zemosaic")
    _set_module_present(monkeypatch, False, product="zemosaic")
    assert analyzer_launch.detect_zemosaic_command() is None
