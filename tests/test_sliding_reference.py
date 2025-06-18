import os
from pathlib import Path
import importlib
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
SeestarQueuedStacker = importlib.import_module("seestar.queuep.queue_manager").SeestarQueuedStacker


def test_sliding_ref_noop():
    q = SeestarQueuedStacker(settings=None)
    q.update_ref_every = 0
    q._update_sliding_reference(np.ones((1, 1), dtype=np.float32))
    assert q.current_ref_image is None
    assert q._images_since_ref == 0


def test_sliding_ref_triggers():
    q = SeestarQueuedStacker(settings=None)
    q.update_ref_every = 40
    for i in range(120):
        img = np.full((1, 1), i, dtype=np.float32)
        q._update_sliding_reference(img)
        if (i + 1) % 40 == 0:
            assert np.array_equal(q.current_ref_image, img)


def test_sliding_ref_skip_on_failure():
    q = SeestarQueuedStacker(settings=None)
    q.update_ref_every = 40
    for i in range(120):
        img = np.full((1, 1), i, dtype=np.float32)
        if i == 24:
            continue
        q._update_sliding_reference(img)
        if i == 40:
            assert np.array_equal(q.current_ref_image, img)


def test_default_disabled():
    q = SeestarQueuedStacker(settings=None)
    assert q.update_ref_every == 0
