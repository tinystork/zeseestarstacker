"""Offscreen Qt tests for the ZeAnalyser reference-return watcher (M25.5-A).

Covers the periodic command-file watcher added to ``MainWindow`` without
touching the scientific backend and without ever spawning a real ZeAnalyser
process:

* launch arms a GUI-thread ``QTimer`` and does not consume a not-yet-present
  reference,
* a ``REFERENCE=<path>`` line arriving later is consumed on the next tick and
  drives the historical Tk consequences (reference field/state, output
  preparation, run start),
* the watcher stops after settlement and is idempotent (one reference consumed
  once, no duplicate GUI updates),
* window close stops the watcher (no zombie ``QTimer`` callback into a
  destroyed ``MainWindow``; the timer is parented to the window),
* no reference yet keeps the watcher running with no spurious update.

``QT_QPA_PLATFORM=offscreen`` is set defensively before any ``QApplication`` is
created, mirroring the other Qt shell tests.  No subprocess is ever spawned:
the injectable ``_analyzer_launcher`` / ``_analyzer_command_file_maker`` seams
are faked, and ticks are driven synchronously through the exposed
``_analyzer_watch_tick`` slot (no real 1 s sleep).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import MainWindow, create_application


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Single process-wide QApplication for the whole session."""
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _make_launching_window(tmp_path):
    """Return a MainWindow whose Analyse launch is fully faked (no spawn)."""
    win = MainWindow()
    win.input_edit.setText(str(tmp_path))
    command_file = tmp_path / "analyzer_stack_command_999.txt"
    win._analyzer_command_file_maker = lambda: str(command_file)
    win._analyzer_launcher = lambda i, l, c: True
    return win, command_file


def _start_spy(monkeypatch, win):
    """Replace ``_on_start`` with a call recorder (no real run started)."""
    started = []
    monkeypatch.setattr(win, "_on_start", lambda: started.append(True))
    return started


# --------------------------------------------------------------------------
# (a) launch arms the watcher; no initial reference
# --------------------------------------------------------------------------
def test_launch_arms_watcher_without_initial_reference(qapp, tmp_path):
    win, command_file = _make_launching_window(tmp_path)
    try:
        win._on_analyse()
        assert win._analyzer_watch_timer.isActive()
        assert win._analyzer_watch_timer.parent() is win
        assert not command_file.exists()
        assert win.reference_edit.text() == ""
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (b) REFERENCE arrives -> tick updates GUI + triggers historical consequence
# --------------------------------------------------------------------------
def test_reference_arrives_tick_updates_gui_and_triggers_consequence(
    qapp, tmp_path, monkeypatch
):
    win, command_file = _make_launching_window(tmp_path)
    started = _start_spy(monkeypatch, win)
    try:
        win._on_analyse()

        ref_file = tmp_path / "ref.fit"
        ref_file.write_text("", encoding="utf-8")
        command_file.write_text(
            f"REFERENCE={ref_file}\nTIMESTAMP=2026-08-21T18:00:00\n",
            encoding="utf-8",
        )

        win._analyzer_watch_tick()

        # Reference field + settings state updated (single-shot seam path).
        assert win.reference_edit.text() == str(ref_file)
        assert win.collect_settings_state().reference_image_path == str(ref_file)
        assert win._reference_origin_hint == "ZEANALYSER_V1"
        # Output preparation (Tk: default output when none set).
        assert win.output_edit.text() == str(tmp_path / "stack_output_analyzer")
        # Historical consequence: run start triggered exactly once.
        assert started == [True]
        # Watcher settled -> stopped.
        assert not win._analyzer_watch_timer.isActive()
        # Command file consumed (best-effort delete).
        assert not command_file.exists()
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (c) watcher stops after settlement and is idempotent
# --------------------------------------------------------------------------
def test_watcher_stops_after_settlement_and_is_idempotent(
    qapp, tmp_path, monkeypatch
):
    win, command_file = _make_launching_window(tmp_path)
    started = _start_spy(monkeypatch, win)
    try:
        win._on_analyse()

        ref_file = tmp_path / "ref.fit"
        ref_file.write_text("", encoding="utf-8")
        command_file.write_text(f"REFERENCE={ref_file}\n", encoding="utf-8")

        win._analyzer_watch_tick()
        assert started == [True]
        assert not win._analyzer_watch_timer.isActive()

        # A second tick must not re-trigger the consequence (file already
        # consumed -> no duplicate GUI update).
        win._analyzer_watch_tick()
        assert started == [True]
        assert win.reference_edit.text() == str(ref_file)
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (d) window close -> no zombie timer
# --------------------------------------------------------------------------
def test_window_close_stops_watcher_no_zombie(qapp, tmp_path):
    win, command_file = _make_launching_window(tmp_path)
    try:
        win._on_analyse()
        timer = win._analyzer_watch_timer
        assert timer.isActive()
        assert timer.parent() is win

        win.shutdown()
        assert not timer.isActive()

        # A reference written after shutdown must not be consumed: no queued
        # timer event may fire (no zombie callback).
        ref_file = tmp_path / "ref.fit"
        ref_file.write_text("", encoding="utf-8")
        command_file.write_text(f"REFERENCE={ref_file}\n", encoding="utf-8")
        qapp.processEvents()

        assert command_file.exists()
        assert win.reference_edit.text() == ""
    finally:
        win.shutdown()


def test_window_destruction_destroys_parented_timer_no_crash(qapp, tmp_path):
    """Deleting the window must destroy the parented timer (no zombie callback)."""
    win, command_file = _make_launching_window(tmp_path)
    ref_file = tmp_path / "ref.fit"
    ref_file.write_text("", encoding="utf-8")

    win._on_analyse()
    timer = win._analyzer_watch_timer
    assert timer.isActive()
    assert timer.parent() is win

    # Write a reference now so a zombie tick WOULD consume it if one fired.
    command_file.write_text(f"REFERENCE={ref_file}\n", encoding="utf-8")

    # Destroy the window (and thus its child QTimer) without a shutdown call.
    win.deleteLater()
    qapp.processEvents()

    # Nothing consumed the reference (no callback into a destroyed window).
    assert command_file.exists()


# --------------------------------------------------------------------------
# (e) no reference yet -> watcher keeps running, no spurious update
# --------------------------------------------------------------------------
def test_no_reference_keeps_watcher_running_no_spurious_update(
    qapp, tmp_path, monkeypatch
):
    win, command_file = _make_launching_window(tmp_path)
    started = _start_spy(monkeypatch, win)
    try:
        win._on_analyse()

        # No command file yet.
        win._analyzer_watch_tick()
        assert win._analyzer_watch_timer.isActive()
        assert started == []
        assert win.reference_edit.text() == ""

        # A command file with TIMESTAMP only (no REFERENCE) is consumed but
        # triggers no consequence; the watcher keeps running.
        command_file.write_text("TIMESTAMP=2026-08-21T18:00:00\n", encoding="utf-8")
        win._analyzer_watch_tick()
        assert win._analyzer_watch_timer.isActive()
        assert started == []
        assert win.reference_edit.text() == ""
        assert not command_file.exists()  # consumed (empty reference)
    finally:
        win.shutdown()
