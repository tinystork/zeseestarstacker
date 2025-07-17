import logging
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub GUI modules to avoid Tk dependence during import
if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = [str(ROOT / "seestar" / "gui")]
    settings_mod = types.ModuleType("seestar.gui.settings")
    settings_mod.SettingsManager = object
    hist_mod = types.ModuleType("seestar.gui.histogram_widget")
    hist_mod.HistogramWidget = object
    gui_pkg.settings = settings_mod
    gui_pkg.histogram_widget = hist_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod
    sys.modules["seestar.gui.histogram_widget"] = hist_mod

from seestar.gui.main_window import SeestarStackerGUI
from seestar.queuep.queue_manager import SeestarQueuedStacker


def test_single_batch_csv(tmp_path):
    # create dummy images
    files = []
    for i in range(3):
        fp = tmp_path / f"img{i}.fits"
        fp.write_text("dummy")
        files.append(fp)

    csv_path = tmp_path / "zenalakyser_order.csv"
    csv_path.write_text("\n".join(f.name for f in files))

    gui = SeestarStackerGUI.__new__(SeestarStackerGUI)
    gui.logger = logging.getLogger("test")
    gui.settings = types.SimpleNamespace(
        input_folder=str(tmp_path),
        batch_size=1,
        stacking_mode="kappa-sigma",
        reproject_between_batches=True,
        use_drizzle=True,
        order_csv_path="",
    )
    gui.queued_stacker = SeestarQueuedStacker()

    activated = SeestarStackerGUI._prepare_single_batch_if_needed(gui)
    assert activated
    assert gui.settings.stacking_mode == "winsorized-sigma"
    assert gui.settings.batch_size == 3
    assert gui.queued_stacker.total_batches_estimated == 1
