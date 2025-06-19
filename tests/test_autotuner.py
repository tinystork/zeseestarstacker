import importlib
import sys
import types
import builtins

# minimal package stubs for queue_manager dependencies
if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    base = str(__file__).split("tests")[0] + "seestar"
    seestar_pkg.__path__ = [base]

    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = []
    settings_mod = types.ModuleType("seestar.gui.settings")
    settings_mod.SettingsManager = object
    gui_pkg.settings = settings_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod

    zmod = types.ModuleType("zemosaic")
    zmod.zemosaic_config = types.SimpleNamespace(
        get_astap_default_search_radius=lambda: 0
    )
    sys.modules.setdefault("zemosaic", zmod)

qm = importlib.import_module("seestar.queuep.queue_manager")


class DummyStacker(qm.SeestarQueuedStacker):
    def __init__(self) -> None:  # type: ignore[override]
        pass


def test_autotune_increase(monkeypatch):
    st = DummyStacker()
    st.thread_fraction = 0.4
    st._configure_global_threads = lambda x: None

    import seestar.queuep.autotuner as at

    monkeypatch.setattr(at, "_PSUTIL_OK", True)

    class DummyPs:
        def cpu_percent(self, interval: int = 1):
            return 20

        def disk_io_counters(self, perdisk: bool = True):
            return {}

    monkeypatch.setitem(builtins.__dict__, "psutil", DummyPs())

    tuner = at.CpuIoAutoTuner(st, duration=2)
    tuner._run()
    assert 0.4 < st.thread_fraction <= 0.75
