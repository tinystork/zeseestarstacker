import types
import importlib
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = []
    settings_mod = types.ModuleType("seestar.gui.settings")
    settings_mod.SettingsManager = object
    gui_pkg.settings = settings_mod
    gui_pkg.histogram_widget = types.ModuleType("seestar.gui.histogram_widget")
    gui_pkg.histogram_widget.HistogramWidget = object
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod
    sys.modules["seestar.gui.histogram_widget"] = gui_pkg.histogram_widget

qm = importlib.import_module("seestar.queuep.queue_manager")


def test_can_resume(tmp_path):
    out = tmp_path
    mem = out / "memmap_accumulators"
    mem.mkdir()
    np.lib.format.open_memmap(
        mem / "cumulative_SUM.npy", mode="w+", dtype=np.float32, shape=(2, 2, 3)
    )
    np.lib.format.open_memmap(
        mem / "cumulative_WHT.npy", mode="w+", dtype=np.float32, shape=(2, 2)
    )
    (out / "batches_count.txt").write_text("1")
    obj = qm.SeestarQueuedStacker()
    assert obj._can_resume(out)
    (out / "batches_count.txt").unlink()
    assert not obj._can_resume(out)


def test_initialize_resume(monkeypatch, tmp_path):
    out = tmp_path
    mem = out / "memmap_accumulators"
    mem.mkdir()
    sum_path = mem / "cumulative_SUM.npy"
    wht_path = mem / "cumulative_WHT.npy"
    np.lib.format.open_memmap(sum_path, mode="w+", dtype=np.float32, shape=(2, 2, 3))[
        :
    ] = 1
    np.lib.format.open_memmap(wht_path, mode="w+", dtype=np.float32, shape=(2, 2))[
        :
    ] = 1
    (out / "batches_count.txt").write_text("2")

    obj = qm.SeestarQueuedStacker()
    obj.resume_mode = True
    obj.output_folder = str(out)
    obj.sum_memmap_path = str(sum_path)
    obj.wht_memmap_path = str(wht_path)
    assert obj.initialize(str(out), (2, 2, 3))
    assert obj.cumulative_sum_memmap is not None
    assert obj.cumulative_sum_memmap.mode == "r+"
