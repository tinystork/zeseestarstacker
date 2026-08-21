"""Qt-local, stdlib-only seam for launching the standalone ZeAnalyser product.

This mirrors the small public-process-contract behaviour of the historical
Tk-side helper (``seestar/gui/analyzer_launch.py``) *without* importing the
Tk GUI or the scientific engine, so :mod:`seestar.gui_qt` stays free of both.

ZeAnalyser is a separately installed/managed ZeSoftware product.  It is
discovered at runtime (it is *not* a declared dependency of ZeSeestarStacker)
through its console entry point ``zeanalyser`` or, failing that, the module
form ``python -m zeanalyser``.

The ZeAnalyser reference-return protocol (env var ``ZEANALYSER_COMMAND_FILE``,
``REFERENCE=``/``TIMESTAMP=`` file format) is a public process contract.  This
module only *consumes* it; it never changes it.

Everything here is stdlib-only, and every side-effecting dependency (executable
detection, process spawning, temp-dir lookup, pid) is injectable so tests never
spawn a real ZeAnalyser process.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Dict, Optional

ANALYZER_ENTRY_POINT = "zeanalyser"
ANALYZER_MODULE = "zeanalyser"
COMMAND_FILE_ENV_VAR = "ZEANALYSER_COMMAND_FILE"
COMM_DIR_NAME = "seestar_stacker_comm"

# The modern ZeAnalyser command-file protocol writes ``REFERENCE=<path>`` and
# ``TIMESTAMP=<...>`` lines.  We only care about the reference here.
_REFERENCE_LINE_RE = re.compile(r"^REFERENCE=(.*)$")


def detect_analyzer_command(which=None, find_spec=None):
    """Return the base launch command for ZeAnalyser, or ``None`` if absent.

    Prefers the ``zeanalyser`` console entry point and falls back to
    ``python -m zeanalyser`` when the module is importable.  ``which`` and
    ``find_spec`` are injectable for tests.
    """
    which_fn = shutil.which if which is None else which
    find_fn = importlib.util.find_spec if find_spec is None else find_spec
    exe = which_fn(ANALYZER_ENTRY_POINT)
    if exe:
        return [exe]
    if find_fn(ANALYZER_MODULE) is not None:
        return [sys.executable, "-m", ANALYZER_MODULE]
    return None


def build_analyzer_command(input_folder, lang, detect=None):
    """Return the full ZeAnalyser launch command, or ``None`` if absent.

    The command carries the input folder to analyse plus the requested UI
    language.  The command-file path is *not* part of the CLI anymore: it is
    passed through the ``ZEANALYSER_COMMAND_FILE`` environment variable.
    """
    detect_fn = detect if detect is not None else detect_analyzer_command
    base = detect_fn()
    if base is None:
        return None
    return base + [
        "--input-dir",
        input_folder,
        "--lang",
        lang,
        "--lock-lang",
    ]


def make_analyzer_env(command_file_path, base_env=None) -> Dict[str, str]:
    """Return a copy of ``base_env`` (default: ``os.environ``) with the
    ``ZEANALYSER_COMMAND_FILE`` variable set to ``command_file_path``."""
    env = dict(base_env if base_env is not None else os.environ)
    env[COMMAND_FILE_ENV_VAR] = command_file_path
    return env


def make_command_file_path(gettempdir=None, getpid=None) -> str:
    """Create and return the command-file path for the current process.

    The directory ``<temp>/seestar_stacker_comm`` is created if missing; the
    file itself is *not* created (ZeAnalyser writes it when it returns a
    reference).  ``gettempdir`` / ``getpid`` are injectable for tests.
    """
    gettempdir_fn = tempfile.gettempdir if gettempdir is None else gettempdir
    getpid_fn = os.getpid if getpid is None else getpid
    app_temp_dir = os.path.join(gettempdir_fn(), COMM_DIR_NAME)
    os.makedirs(app_temp_dir, exist_ok=True)
    return os.path.join(app_temp_dir, f"analyzer_stack_command_{getpid_fn()}.txt")


def launch_analyzer(
    input_folder: str,
    lang: str,
    command_file_path: str,
    popen: Optional[Callable] = None,
    detect: Optional[Callable] = None,
) -> bool:
    """Detect, build, and launch ZeAnalyser as a non-blocking subprocess.

    Returns ``True`` when a process was spawned, or ``False`` when ZeAnalyser
    could not be found (nothing launched).  ``popen`` / ``detect`` are
    injectable for tests; ``popen`` defaults to :func:`subprocess.Popen`.
    """
    cmd = build_analyzer_command(input_folder, lang, detect=detect)
    if cmd is None:
        return False
    env = make_analyzer_env(command_file_path)
    popen_fn = subprocess.Popen if popen is None else popen
    popen_fn(cmd, env=env)
    return True


def parse_reference_from_command_file(content: str) -> Optional[str]:
    """Extract the recommended reference path from command-file content.

    The modern ZeAnalyser command-file protocol writes ``REFERENCE=<path>``
    and ``TIMESTAMP=<...>`` lines.  Returns the reference path, or ``None``
    when no ``REFERENCE=`` line is present.  ``TIMESTAMP=`` and any other
    lines are ignored.
    """
    for raw_line in content.splitlines():
        line = raw_line.strip()
        match = _REFERENCE_LINE_RE.match(line)
        if match:
            ref = match.group(1).strip()
            return ref or None
    return None


def consume_command_file(path: str) -> Optional[str]:
    """Read the command file, delete it (best-effort), return the reference.

    Returns the reference path (str) or ``None``.  The file is deleted
    immediately after a successful read; a deletion failure does not lose the
    already-read reference (matching the historical surveillance-loop
    behaviour).
    """
    with open(path, "r", encoding="utf-8") as f_cmd:
        content = f_cmd.read()
    ref = parse_reference_from_command_file(content)
    try:
        os.remove(path)
    except OSError:
        # File already gone or not removable; content was read successfully.
        pass
    return ref
