import types
import sys
from pathlib import Path

# Ensure seestar package path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seestar.queuep.queue_manager import get_batches_from_stack_plan


def test_get_batches_from_stack_plan(tmp_path):
    plan = tmp_path / "stack_plan.csv"
    (tmp_path / "img1.fits").write_text("a")
    (tmp_path / "img2.fits").write_text("b")
    (tmp_path / "img3.fits").write_text("c")
    plan.write_text(
        "order,batch_id,file_path\n"
        "1,batchA,img1.fits\n"
        "2,batchA,img2.fits\n"
        "3,batchB,img3.fits\n"
    )
    batches = get_batches_from_stack_plan(str(plan), str(tmp_path))
    assert batches == [
        [str(tmp_path / "img1.fits"), str(tmp_path / "img2.fits")],
        [str(tmp_path / "img3.fits")],
    ]
