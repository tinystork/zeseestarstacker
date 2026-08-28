import importlib

qm = importlib.import_module("seestar.queuep.queue_manager")


def test_quality_pool_size(monkeypatch):
    monkeypatch.setattr(qm.os, "cpu_count", lambda: 8)
    assert qm._suggest_pool_size(0.75) == 6
