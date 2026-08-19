"""Launch helpers for the standalone ZeAnalyser product.

ZeAnalyser is a separately installed/managed ZeSoftware product.  It is
discovered at runtime (it is *not* a declared dependency of
ZeSeestarStacker) through its console entry point ``zeanalyser`` or,
failing that, the module form ``python -m zeanalyser``.

The ZeAnalyser reference-return protocol (env var
``ZEANALYSER_COMMAND_FILE``, ``REFERENCE=``/``TIMESTAMP=`` file format) is a
public process contract documented in the ZeAnalyser repository:
``docs/zeanalyser_process_contract.md`` (protocol v1, locked by
``tests/test_process_contract.py``).

This module deliberately avoids importing OpenCV / Pillow / Tkinter so it can
be exercised in minimal environments (the test suite loads it by file path,
mirroring the pattern already used by ``tests/test_solver_config.py``).
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys

ANALYZER_ENTRY_POINT = "zeanalyser"
ANALYZER_MODULE = "zeanalyser"
COMMAND_FILE_ENV_VAR = "ZEANALYSER_COMMAND_FILE"

# The modern ZeAnalyser command-file protocol writes ``REFERENCE=<path>`` and
# ``TIMESTAMP=<...>`` lines.  We only care about the reference here.
_REFERENCE_LINE_RE = re.compile(r"^REFERENCE=(.*)$")


def detect_analyzer_command():
    """Return the base launch command for ZeAnalyser, or ``None`` if absent.

    Prefers the ``zeanalyser`` console entry point and falls back to
    ``python -m zeanalyser`` when the module is importable.
    """
    exe = shutil.which(ANALYZER_ENTRY_POINT)
    if exe:
        return [exe]
    if importlib.util.find_spec(ANALYZER_MODULE) is not None:
        return [sys.executable, "-m", ANALYZER_MODULE]
    return None


def build_analyzer_command(input_folder, lang):
    """Return the full ZeAnalyser launch command, or ``None`` if absent.

    The command carries the input folder to analyse plus the requested UI
    language.  The command-file path is *not* part of the CLI anymore: it is
    passed through the ``ZEANALYSER_COMMAND_FILE`` environment variable.
    """
    base = detect_analyzer_command()
    if base is None:
        return None
    return base + [
        "--input-dir",
        input_folder,
        "--lang",
        lang,
        "--lock-lang",
    ]


def make_analyzer_env(command_file_path, base_env=None):
    """Return a copy of ``base_env`` (default: ``os.environ``) with the
    ``ZEANALYSER_COMMAND_FILE`` variable set to ``command_file_path``."""
    env = dict(base_env if base_env is not None else os.environ)
    env[COMMAND_FILE_ENV_VAR] = command_file_path
    return env


def launch_analyzer(input_folder, lang, command_file_path, popen=None):
    """Detect, build, and launch ZeAnalyser as a non-blocking subprocess.

    Returns ``True`` when a process was spawned, or ``False`` when ZeAnalyser
    could not be found (nothing launched).  ``popen`` is injectable for tests;
    it defaults to :func:`subprocess.Popen`.
    """
    cmd = build_analyzer_command(input_folder, lang)
    if cmd is None:
        return False
    env = make_analyzer_env(command_file_path)
    popen_fn = subprocess.Popen if popen is None else popen
    popen_fn(cmd, env=env)
    return True


def parse_reference_from_command_file(content):
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


def consume_command_file(path):
    """Read the command file, delete it, and return the parsed reference.

    Returns the reference path (str) or ``None``.  The file is deleted
    immediately after a successful read (best-effort: a deletion failure does
    not lose the already-read reference), matching the previous behaviour of
    the analyser command-file surveillance loop.
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
