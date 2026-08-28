import importlib
import numpy as np

qm = importlib.import_module("seestar.queuep.queue_manager")

class DummyExecutor:
    created = 0
    def __init__(self, *a, **k):
        DummyExecutor.created += 1
        self._max_workers = k.get("max_workers") if k else (a[0] if a else None)
        self._shutdown = False
    class DummyFuture:
        def __init__(self, res):
            self._res = res
        def result(self):
            return self._res
    def submit(self, fn, *a, **k):
        return DummyExecutor.DummyFuture(fn(*a, **k))
    def shutdown(self, wait=True, cancel_futures=False):
        self._shutdown = True

def dummy_worker(data):
    return {"snr": 2.0, "stars": 1.0}, None, 1


def test_quality_executor_recreate(monkeypatch):
    monkeypatch.setattr(qm, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(qm, "_quality_metrics_worker", dummy_worker)
    s = qm.SeestarQueuedStacker()
    base = DummyExecutor.created
    first = s._calculate_quality_metrics(np.zeros((1, 1), dtype=np.float32))
    s.quality_executor.shutdown()
    second = s._calculate_quality_metrics(np.zeros((1, 1), dtype=np.float32))
    assert first["snr"] > 0 and second["snr"] > 0
    assert DummyExecutor.created == base + 1
    s.__class__.stop_processing(s)
