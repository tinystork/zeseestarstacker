"""M25 boring ``--max-mem`` delta tests.

The boring (single-batch CSV) route used to hardcode ``max_mem_gb=8.0``, so the
user-configured "HQ RAM limit (GB)" (``QtSettingsState.max_hq_mem_gb``) was
ignored by the subprocess launch even though the regular run path already
forwarded it (M20 seam).  The Tk boring branch *does* forward the configured
value::

    "--max-mem", str(getattr(self.settings, "max_hq_mem_gb", 8)),

where ``self.settings.max_hq_mem_gb`` is a float read from the ``max_hq_mem_var``
``tk.DoubleVar`` (default ``8.0``).  So the Qt hardcode was a deviation from Tk,
not a parity match.  This lot wires ``max_hq_mem_gb`` into the boring request
while preserving the ``8.0`` default for callers that pass nothing.

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
# (1) default: no memory value provided -> 8.0 fallback preserved
# --------------------------------------------------------------------------
def test_build_boring_request_default_max_mem_fallback():
    req = build_boring_request(
        csv_path="/in/stack_plan.csv",
        output_dir="/out",
        python_executable="/usr/bin/python3",
    )
    assert req.max_mem_gb == 8.0
    assert _arg_after(req.command, "--max-mem") == "8.0"
    # The structured field matches the argv value exactly.
    assert _arg_after(req.command, "--max-mem") == str(req.max_mem_gb)


def test_boring_route_default_window_forwards_8_0(qapp, tmp_path):
    """A bare window (untouched HQ-RAM spin) forwards the 8.0 default."""
    win, fakes = _prepare(tmp_path)
    try:
        req = _start_and_get_request(win, fakes)
        assert req.max_mem_gb == 8.0
        assert _arg_after(req.command, "--max-mem") == "8.0"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (2) settings provide a value -> argv contains that value
# --------------------------------------------------------------------------
def test_boring_route_forwards_configured_max_mem(qapp, tmp_path):
    win, fakes = _prepare(tmp_path)
    try:
        win.max_hq_mem_spin.setValue(4)
        req = _start_and_get_request(win, fakes)
        assert req.max_mem_gb == 4.0
        assert _arg_after(req.command, "--max-mem") == "4.0"
    finally:
        win.shutdown()


def test_boring_route_forwards_configured_max_mem_12(qapp, tmp_path):
    win, fakes = _prepare(tmp_path)
    try:
        win.max_hq_mem_spin.setValue(12)
        req = _start_and_get_request(win, fakes)
        assert req.max_mem_gb == 12.0
        assert _arg_after(req.command, "--max-mem") == "12.0"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (3) parity statement: Qt boring argv matches the Tk reference
# --------------------------------------------------------------------------
def test_boring_max_mem_matches_tk_reference(qapp, tmp_path):
    """Qt boring ``--max-mem`` == the Tk boring branch, value-for-value.

    The Tk reference is ``str(getattr(self.settings, "max_hq_mem_gb", 8))``
    with ``max_hq_mem_gb`` a float (default ``8.0``): Tk *always* forwards the
    configured HQ-RAM value and never omits ``--max-mem``.  Qt must forward
    ``float(state.max_hq_mem_gb)`` (default ``8.0``) — asserted here for both
    the default and a configured value.
    """
    # Default case == Tk default 8.0 (float).
    win1, fakes1 = _prepare(tmp_path)
    try:
        req = _start_and_get_request(win1, fakes1)
        assert _arg_after(req.command, "--max-mem") == str(float(8))
    finally:
        win1.shutdown()

    # Configured case is forwarded identically to Tk (float formatting).
    win2, fakes2 = _prepare(tmp_path)
    try:
        win2.max_hq_mem_spin.setValue(16)
        req = _start_and_get_request(win2, fakes2)
        assert _arg_after(req.command, "--max-mem") == str(float(16))
        assert _arg_after(req.command, "--max-mem") == "16.0"
    finally:
        win2.shutdown()


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
