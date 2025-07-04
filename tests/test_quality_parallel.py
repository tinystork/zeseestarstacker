import importlib
import sys
import types
import concurrent.futures as cf
import time

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
from tests import dummy_quality_worker as dq


class DummyStacker:
    _calculate_quality_metrics = qm.SeestarQueuedStacker._calculate_quality_metrics

    def __init__(self, workers: int):
        self.quality_executor = cf.ProcessPoolExecutor(max_workers=workers)

    def update_progress(self, *a, **k):
        pass


def _run(stacker: DummyStacker, n: int = 20) -> float:
    import numpy as np

    imgs = [np.zeros((10, 10), dtype=np.float32) for _ in range(n)]
    start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(stacker._calculate_quality_metrics, img) for img in imgs]
        for f in futures:
            f.result()
    duration = time.perf_counter() - start
    return duration


def test_quality_parallel(monkeypatch):
    monkeypatch.setattr(qm, "_quality_metrics_worker", dq.dummy_worker)
    monkeypatch.setattr(qm.os, "cpu_count", lambda: 8)
    fast_workers = qm._suggest_pool_size(0.75)

    fast = DummyStacker(fast_workers)
    slow = DummyStacker(1)
    t_fast = _run(fast)
    t_slow = _run(slow)
    fast.quality_executor.shutdown()
    slow.quality_executor.shutdown()
    assert t_slow / t_fast >= 3
