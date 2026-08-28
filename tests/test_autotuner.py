import importlib
import builtins

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
