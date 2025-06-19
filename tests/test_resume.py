import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

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
