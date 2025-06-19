import importlib.util
import sys
import types
from pathlib import Path
import os

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# minimal stubs for GUI modules
if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = []
    settings_mod = types.ModuleType("seestar.gui.settings")

    class DummySettingsManager:
        pass

    settings_mod.SettingsManager = DummySettingsManager
    gui_pkg.settings = settings_mod
    sys.modules["seestar.gui.settings"] = settings_mod
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg

from seestar.queuep.queue_manager import SeestarQueuedStacker


def test_can_resume(tmp_path):
    out = tmp_path
    memdir = out / "memmap_accumulators"
    memdir.mkdir()
    np.lib.format.open_memmap(
        memdir / "cumulative_SUM.npy", mode="w+", dtype=np.float32, shape=(2, 2, 3)
    )[:]
    np.lib.format.open_memmap(
        memdir / "cumulative_WHT.npy", mode="w+", dtype=np.float32, shape=(2, 2)
    )[:]
    (out / "batches_count.txt").write_text("2")
    s = SeestarQueuedStacker()
    assert s._can_resume(out)


def test_open_existing_memmaps(tmp_path):
    out = tmp_path
    memdir = out / "memmap_accumulators"
    memdir.mkdir()
    np.lib.format.open_memmap(
        memdir / "cumulative_SUM.npy", mode="w+", dtype=np.float32, shape=(2, 2, 3)
    )[:]
    np.lib.format.open_memmap(
        memdir / "cumulative_WHT.npy", mode="w+", dtype=np.float32, shape=(2, 2)
    )[:]
    count_path = out / "batches_count.txt"
    count_path.write_text("3")

    s = SeestarQueuedStacker()
    s.output_folder = str(out)
    s.batch_count_path = str(count_path)
    assert s._open_existing_memmaps()
    assert s.memmap_shape == (2, 2, 3)
    assert s.stacked_batches_count == 3



def test_save_partial_stack(tmp_path):
    out = tmp_path
    s = SeestarQueuedStacker()
    s.output_folder = str(out)
    s.output_filename = "stack"
    s.cumulative_sum_memmap = np.zeros((2, 2, 3), dtype=np.float32)
    s.cumulative_wht_memmap = np.ones((2, 2), dtype=np.float32)
    s.stacked_batches_count = 2
    s.partial_save_interval = 1

    class DummyVar:
        def set(self, value):
            self.value = value

    s.gui = types.SimpleNamespace(last_stack_path=DummyVar())

    # create previous intermediate stack to ensure it's removed
    prev = out / "stack_batch001.fit"
    prev.write_bytes(b"test")

    s._save_partial_stack()

    expected = out / "stack_batch002.fit"
    assert expected.exists()
    assert not prev.exists()


def test_save_partial_stack_failure_keeps_previous(tmp_path, monkeypatch):
    out = tmp_path
    s = SeestarQueuedStacker()
    s.output_folder = str(out)
    s.output_filename = "stack"
    s.cumulative_sum_memmap = np.zeros((2, 2, 3), dtype=np.float32)
    s.cumulative_wht_memmap = np.ones((2, 2), dtype=np.float32)
    s.stacked_batches_count = 2
    s.partial_save_interval = 1

    prev = out / "stack_batch001.fit"
    prev.write_bytes(b"test")

    def fail_replace(src, dst):
        raise RuntimeError("boom")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(RuntimeError):
        s._save_partial_stack()

    assert prev.exists()

