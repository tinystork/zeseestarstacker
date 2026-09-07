"""Stage F — early CLI ``--version`` / ``-V`` for the Qt entry point.

8.4.0 stage F contract: ``python -m seestar.qt_main --version`` (or the
``zeseestarstacker`` console script, or ``main(["--version"])``) prints a
stable one-line product identity containing the package version (8.4.0) and
codename, returns exit code 0, and NEVER constructs the Qt application or
launches the engine — the heavy Qt shell is only imported lazily inside
``main`` for a real launch.

Import hygiene: ``seestar.qt_main`` itself stays Qt-free at module import time
(after the stage F lazy-import change), so the in-process tests below never
pull in PySide6.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _pkg_version() -> str:
    text = (ROOT / "seestar" / "__init__.py").read_text(encoding="utf-8")
    import re

    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    assert m
    return m.group(1)


def test_qt_main_module_imports_without_qt():
    """The module-level import stays Qt-free (lazy run_qt_app import):
    importing ``seestar.qt_main`` must not newly pull in any PySide6 module
    (other tests in this process may already have imported Qt — we only pin
    that THIS module adds none)."""
    before = set(sys.modules)
    import seestar.qt_main as qt_main  # noqa: F401

    new_modules = set(sys.modules) - before
    qt_modules = [m for m in new_modules if m.split(".")[0] == "PySide6"]
    assert not qt_modules, f"qt_main import pulled in Qt: {qt_modules}"
    assert hasattr(qt_main, "main")
    assert hasattr(qt_main, "parse_qt_args")


def test_main_version_prints_version_and_exit_zero(capsys):
    import seestar.qt_main as qt_main

    code = qt_main.main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "ZeSeestarStacker" in out
    assert _pkg_version() in out
    assert "Phoenix" in out


def test_main_dash_capital_v_also_prints_version(capsys):
    import seestar.qt_main as qt_main

    code = qt_main.main(["-V"])
    out = capsys.readouterr().out
    assert code == 0
    assert _pkg_version() in out


def test_main_version_wins_over_backend_tokens(capsys):
    """--version is handled before backend resolution; unknown/extra tokens
    do not matter and no backend is resolved."""
    import seestar.qt_main as qt_main

    code = qt_main.main(["--version", "--backend", "simulated", "--whatever"])
    out = capsys.readouterr().out
    assert code == 0
    assert _pkg_version() in out


def test_no_qapplication_constructed_for_version(capsys):
    """--version must never construct the Qt application (no PySide import)."""
    # Isolate from any prior Qt import in this process by checking the marker:
    # main(["--version"]) must return without importing PySide6.QtWidgets.
    import seestar.qt_main as qt_main

    before = "PySide6.QtWidgets" in sys.modules
    code = qt_main.main(["--version"])
    capsys.readouterr()
    assert code == 0
    if not before:
        assert "PySide6.QtWidgets" not in sys.modules, (
            "--version must not construct/import the Qt application"
        )


def test_subprocess_python_m_version():
    """``python -m seestar.qt_main --version`` exits 0 and prints 8.4.0."""
    env = dict(os.environ)
    env.pop("ZSSS_QT_STARTUP_WITNESS", None)
    proc = subprocess.run(
        [sys.executable, "-m", "seestar.qt_main", "--version"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert f"ZeSeestarStacker {_pkg_version()}" in proc.stdout
    assert _pkg_version() in proc.stdout
