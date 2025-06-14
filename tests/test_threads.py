import importlib
import sys
import types

if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str((__file__)).split("tests")[0] + "seestar"]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = []
    settings_mod = types.ModuleType("seestar.gui.settings")
    settings_mod.SettingsManager = object
    gui_pkg.settings = settings_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod

qm = importlib.import_module("seestar.queuep.queue_manager")


def test_configure_threads(monkeypatch):
    records = {}

    def fake_limits(n):
        records["tp"] = n
    monkeypatch.setattr(qm, "threadpoolctl", types.SimpleNamespace(threadpool_limits=fake_limits))
    monkeypatch.setattr(qm.cv2, "setNumThreads", lambda n: records.setdefault("cv2", n))
    monkeypatch.setitem(sys.modules, "mkl", types.SimpleNamespace(set_num_threads=lambda n: records.setdefault("mkl", n)))
    obj = qm.SeestarQueuedStacker(thread_fraction=0.5)
    import os
    assert obj.num_threads == max(1, int(os.cpu_count() * 0.5))
    assert records["tp"] == obj.num_threads
    assert records["cv2"] == obj.num_threads
    assert records["mkl"] == obj.num_threads


def test_process_batch_parallel(monkeypatch):
    obj = qm.SeestarQueuedStacker(thread_fraction=0.5)
    called = []

    def fake_process(path):
        called.append(path)
        return path + "_done"

    obj._process_file = fake_process

    class DummyExec:
        def __init__(self, max_workers):
            DummyExec.workers = max_workers
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def map(self, fn, iterable):
            return [fn(x) for x in iterable]

    import concurrent.futures as cf
    monkeypatch.setattr(cf, "ThreadPoolExecutor", DummyExec)

    res = obj._process_batch_parallel(["a", "b"])
    assert res == ["a_done", "b_done"]
    assert DummyExec.workers == obj.num_threads
