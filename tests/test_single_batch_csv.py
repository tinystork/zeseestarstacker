import logging
import sys
import types
import os
import numpy as np
import pytest
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
    settings_mod.TILE_HEIGHT = 512
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

    csv_path = tmp_path / "stack_plan.csv"

    csv_path.write_text("order,file\n" + "\n".join(f.name for f in files))


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
    assert gui.settings.batch_size == 1
    assert gui.settings.order_csv_path == str(csv_path)


def test_single_batch_csv_with_index(tmp_path):
    files = []
    for i in range(3):
        fp = tmp_path / f"img{i}.fits"
        fp.write_text("dummy")
        files.append(fp)

    csv_path = tmp_path / "stack_plan.csv"
    lines = ["index,file"]
    for i, f in enumerate(files, start=1):
        lines.append(f"{i},{f.name}")
    csv_path.write_text("\n".join(lines))

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
    assert gui.settings.batch_size == 1
    assert gui.settings.order_csv_path == str(csv_path)


def test_single_batch_csv_with_additional_columns(tmp_path):
    files = []
    for i in range(3):
        fp = tmp_path / f"img{i}.fits"
        fp.write_text("dummy")
        files.append(fp)

    csv_path = tmp_path / "stack_plan.csv"
    header = (
        "order,batch_id,mount,bortle,telescope,session_date,filter,exposure,file_path"
    )
    lines = [header]
    for i, f in enumerate(files, start=1):
        lines.append(
            f"{i},1,m1,3,ts,2024-01-01,RGB,10.0,{f.name}"
        )
    csv_path.write_text("\n".join(lines))

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
    assert gui.settings.batch_size == 1
    assert gui.settings.order_csv_path == str(csv_path)


def test_single_batch_csv_missing_file(tmp_path):
    """Missing ``stack_plan.csv`` should raise an error when ``batch_size`` is 1."""

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

    with pytest.raises(FileNotFoundError):
        SeestarStackerGUI._prepare_single_batch_if_needed(gui)


def test_align_on_disk_large_image(tmp_path):
    from seestar.core.alignment import SeestarAligner
    img = np.random.rand(100, 100, 3).astype(np.float32)
    ref = img.copy()
    aligner = SeestarAligner()
    import astroalign as aa
    import cv2

    def dummy_find_transform(*args, **kwargs):
        from skimage.transform import SimilarityTransform
        return SimilarityTransform(), ([], [])

    orig = aa.find_transform
    aa.find_transform = dummy_find_transform
    orig_warp = aligner._align_cpu
    def dummy_align_cpu(self, img, M, dsize, out=None):
        if out is None:
            return img.copy()
        out[:] = img
        return out
    aligner._align_cpu = dummy_align_cpu.__get__(aligner, SeestarAligner)
    try:
        aligned, success = aligner._align_image(img, ref, "x", use_disk=True)
        assert success
        assert isinstance(aligned, np.memmap)
    finally:
        aa.find_transform = orig
        aligner._align_cpu = orig_warp
        if isinstance(aligned, np.memmap):
            os.remove(aligned.filename)


def test_threaded_csv_no_chunk(tmp_path):
    files = []
    for i in range(3):
        fp = tmp_path / f"img{i}.fits"
        fp.write_text("dummy")
        files.append(fp)

    csv_path = tmp_path / "stack_plan.csv"
    csv_path.write_text("file_path\n" + "\n".join(f.name for f in files))

    gui = SeestarStackerGUI.__new__(SeestarStackerGUI)
    gui.logger = logging.getLogger("test")
    gui.settings = types.SimpleNamespace(
        input_folder=str(tmp_path),
        output_folder=str(tmp_path),
        temp_folder=str(tmp_path),
        output_filename="out",
        reference_image_path="",
        batch_size=1,
        stacking_mode="kappa-sigma",
        reproject_between_batches=True,
        use_drizzle=True,
        order_csv_path="",
    )
    gui.queued_stacker = types.SimpleNamespace()
    captured = {}

    def fake_start(**kw):
        captured.update(kw)
        return True

    gui.queued_stacker.start_processing = fake_start

    activated = SeestarStackerGUI._prepare_single_batch_if_needed(gui)
    assert activated

    kwargs = {
        "batch_size": gui.settings.batch_size,
        "ordered_files": gui.settings.order_file_list,
    }
    if gui.settings.batch_size == 1 and not activated:
        kwargs["chunk_size"] = 99

    gui.queued_stacker.start_processing(**kwargs)

    assert "chunk_size" not in captured
    assert captured["ordered_files"] == [str(p) for p in files]
