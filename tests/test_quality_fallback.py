import importlib
import numpy as np

qm = importlib.import_module("seestar.queuep.queue_manager")


def dummy_worker(data):
    return {"snr": 1.0, "stars": 1.0}, None, 1


def test_quality_fallback_large(monkeypatch):
    monkeypatch.setattr(qm, "_quality_metrics_worker", dummy_worker)
    s = qm.SeestarQueuedStacker()
    # create >32 MB array
    big = np.zeros((4096, 4096), dtype=np.float32)
    res = s._calculate_quality_metrics(big)
    assert res["snr"] == 1.0
    s.__class__.stop_processing(s)
