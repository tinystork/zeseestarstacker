"""M25 boring ``--max-mem`` delta tests (8.4.0 stage E2 migration).

The boring (single-batch CSV) route historically forwarded the user-configured
"HQ RAM limit (GB)" as ``--max-mem`` (Qt M25 wiring / Tk ``boring_stack``
branch).  8.4.0 stage E2 removes HQ RAM as a user-facing scientific control:
AUTO is the product CPU memory policy, so a NORMAL boring launch does NOT
forward any memory value and the subprocess resolves its own automatic policy.
Only an EXPLICIT expert override (``build_boring_request(max_mem_gb=...)``,
CI/test/debug/RAM simulation) emits ``--max-mem``; the boring subprocess turns
it into the provenance-visible OVERRIDE env seam
(``ZSSS_CPU_MEMORY_OVERRIDE_BYTES``), never a silent default.

No subprocess is ever spawned: every window under test injects a fake runner
and asserts the argv built into the ``BoringRunRequest``.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.boring_route import (
    BoringRunRequest,
    build_boring_request,
    resolve_boring_script_path,
)
from seestar.gui_qt.boring_runner import BoringRunnerBase


class FakeBoringRunner(BoringRunnerBase):
    """Records the request without launching a real subprocess."""

    def __init__(self) -> None:
        super().__init__()
        self.start_calls = []
        self._active = False

    def start(self, request: BoringRunRequest) -> None:
        self.start_calls.append(request)
        self._active = True
        self.started.emit()

    def cancel(self) -> None:
        self._active = False

    def is_running(self) -> bool:
        return self._active


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _arg_after(cmd, name):
    return cmd[cmd.index(name) + 1]


def _prepare(tmp_path: Path):
    """Create an input folder with a valid ``stack_plan.csv`` + fake runner."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(exist_ok=True)
    (input_dir / "a.fits").write_bytes(b"")
    with open(input_dir / "stack_plan.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["a.fits"])
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)

    fakes = []

    def factory():
        fake = FakeBoringRunner()
        fakes.append(fake)
        return fake

    win = MainWindow(boring_runner_factory=factory)
    win.input_edit.setText(str(input_dir))
    win.output_edit.setText(str(output_dir))
    win.batch_spin.setValue(1)
    return win, fakes


def _start_and_get_request(win, fakes) -> BoringRunRequest:
    win.start_button.click()
    assert len(fakes) == 1
    return fakes[0].start_calls[-1]


# --------------------------------------------------------------------------
# (1) normal boring: no memory value forwarded -> AUTO (--max-mem omitted)
# --------------------------------------------------------------------------
def test_build_boring_request_default_omits_max_mem():
    req = build_boring_request(
        csv_path="/in/stack_plan.csv",
        output_dir="/out",
        python_executable="/usr/bin/python3",
    )
    assert req.max_mem_gb is None
    assert "--max-mem" not in req.command


def test_boring_route_default_window_omits_max_mem(qapp, tmp_path):
    """A bare window (normal product policy) launches boring under AUTO — no
    HQ-RAM value is forwarded to the subprocess."""
    win, fakes = _prepare(tmp_path)
    try:
        req = _start_and_get_request(win, fakes)
        assert req.max_mem_gb is None
        assert "--max-mem" not in req.command
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (2) env override never leaks into argv (it is inherited by the subprocess
#     and read by the engine as the provenance-visible OVERRIDE seam)
# --------------------------------------------------------------------------
def test_boring_route_env_override_does_not_add_argv_flag(qapp, tmp_path, monkeypatch):
    win, fakes = _prepare(tmp_path)
    try:
        monkeypatch.setenv("ZSSS_CPU_MEMORY_OVERRIDE_BYTES", str(4 * 1024 ** 3))
        req = _start_and_get_request(win, fakes)
        # The explicit override travels through the environment (inherited by
        # the boring subprocess), never through a normal GUI argv flag.
        assert req.max_mem_gb is None
        assert "--max-mem" not in req.command
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (3) explicit expert override still forwards --max-mem
# --------------------------------------------------------------------------
def test_boring_route_explicit_override_forwards_max_mem(qapp, tmp_path):
    """An EXPLICIT expert override (CI/test/debug/RAM simulation) still emits
    ``--max-mem``; the boring subprocess turns it into the provenance-visible
    OVERRIDE seam — never a silent default."""
    win, fakes = _prepare(tmp_path)
    try:
        req = build_boring_request(
            csv_path=str(tmp_path / "inputs" / "stack_plan.csv"),
            output_dir=str(tmp_path / "outputs"),
            max_mem_gb=16.0,
            python_executable="/py",
        )
        assert req.max_mem_gb == 16.0
        assert _arg_after(req.command, "--max-mem") == "16.0"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Command shape is unchanged apart from the wired memory value
# --------------------------------------------------------------------------
def test_boring_command_shape_unchanged_with_memory():
    req = build_boring_request(
        csv_path="/in/stack_plan.csv",
        output_dir="/out",
        batch_size=1,
        chunk_size=77,
        normalize_method="none",
        save_final_as_float32=False,
        final_combine="mean",
        max_mem_gb=4.0,
        python_executable="/py",
    )
    cmd = req.command
    assert cmd[0] == "/py"
    assert cmd[1] == resolve_boring_script_path()
    assert _arg_after(cmd, "--csv") == "/in/stack_plan.csv"
    assert _arg_after(cmd, "--out") == "/out"
    assert _arg_after(cmd, "--batch-size") == "1"
    assert _arg_after(cmd, "--max-mem") == "4.0"
    assert _arg_after(cmd, "--chunk-size") == "77"
    assert _arg_after(cmd, "--norm") == "none"
    assert "--no-save-as-float32" in cmd
    assert _arg_after(cmd, "--final-combine") == "mean"
