import importlib
import sys
import types
import threading
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub GUI modules to avoid Tk dependence during import
if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = [str(ROOT / "seestar" / "gui")]
    settings_mod = types.ModuleType("seestar.gui.settings")
    settings_mod.SettingsManager = object
    settings_mod.TILE_HEIGHT = 512
    hist_mod = types.ModuleType("seestar.gui.histogram_widget")
    hist_mod.HistogramWidget = object
    gui_pkg.settings = settings_mod
    gui_pkg.histogram_widget = hist_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod
    sys.modules["seestar.gui.histogram_widget"] = hist_mod

    zmod = types.ModuleType("zemosaic")
    zmod.zemosaic_config = types.SimpleNamespace(
        get_astap_default_search_radius=lambda: 0
    )
    sys.modules.setdefault("zemosaic", zmod)

mw = importlib.import_module("seestar.gui.main_window")
bs = importlib.import_module("seestar.gui.boring_stack")
import tkinter as tk


class DummyWidget:
    def __init__(self):
        self.states = []

    def config(self, **kw):
        if "state" in kw:
            self.states.append(kw["state"])

    def winfo_exists(self):
        return True


class DummyProc:
    def __init__(self, *a, **k):
        self.stdout = iter(["100%"])

    def wait(self):
        return 0


def test_boring_thread_starts_thread(monkeypatch, tmp_path):
    gui = mw.SeestarStackerGUI.__new__(mw.SeestarStackerGUI)
    gui.root = types.SimpleNamespace(after=lambda *a, **k: a[1](*a[2:]))
    gui.start_button = DummyWidget()
    gui.stop_button = DummyWidget()
    gui._set_parameter_widgets_state = lambda *a, **k: None
    gui.update_progress_gui = lambda *a, **k: None
    gui.progress_manager = None
    monkeypatch.setattr(mw, "messagebox", types.SimpleNamespace(showerror=lambda *a, **k: None, showinfo=lambda *a, **k: None))

    monkeypatch.setattr(subprocess, "Popen", DummyProc)
    monkeypatch.setattr(bs, "read_paths", lambda p: ["a"])

    created = []

    class DummyThread:
        def __init__(self, target=None, args=(), kwargs=None, **kw):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

        def run(self):
            self.target(*self.args, **self.kwargs)

    monkeypatch.setattr(threading, "Thread", DummyThread)

    csv_path = tmp_path / "plan.csv"
    csv_path.write_text("file\nimg.fits")
    out_dir = tmp_path
    cmd = [sys.executable, "boring_stack.py", "--csv", str(csv_path), "--out", str(out_dir)]

    gui._run_boring_stack_process(cmd, str(csv_path), str(out_dir))

    assert created and created[0].started
    # worker did not run yet
    assert gui.start_button.states == []

    created[0].run()

    assert gui.start_button.states == [tk.DISABLED, tk.NORMAL]
    assert gui.stop_button.states == [tk.NORMAL, tk.DISABLED]
