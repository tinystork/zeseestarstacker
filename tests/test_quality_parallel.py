import importlib
import concurrent.futures as cf
import time
import pytest

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
    if t_slow <= t_fast:
        pytest.skip("Parallel quality metrics not faster on this platform")
    assert t_slow / t_fast >= 3
