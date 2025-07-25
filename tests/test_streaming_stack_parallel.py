import os
import numpy as np
from astropy.io import fits

import seestar.core.streaming_stack as ss


def test_parallel_io_settings(monkeypatch, tmp_path):
    files = []
    for i in range(3):
        arr = np.full((4, 4), i + 1, dtype=np.float32)
        p = tmp_path / f"img{i}.fits"
        fits.PrimaryHDU(arr).writeto(p)
        files.append(str(p))

    called = {}

    def fake_estimate(num_files, shape_hw, max_ram_frac=0.3, max_workers_cap=8):
        called["args"] = (num_files, shape_hw, max_ram_frac, max_workers_cap)
        return 2, 3

    monkeypatch.setattr(ss, "estimate_parallel_io_settings", fake_estimate)

    workers = {}

    class DummyExecutor:
        def __init__(self, max_workers):
            workers["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def map(self, func, iterable):
            return [func(x) for x in iterable]

    monkeypatch.setattr(ss, "ThreadPoolExecutor", DummyExecutor)

    monkeypatch.chdir(tmp_path)
    out = ss.stack_disk_streaming(files, chunk_rows=0, parallel_io=True)

    assert called["args"][0] == len(files)
    assert workers["max_workers"] == 3

    with fits.open(out, memmap=True) as hdul:
        data = hdul[0].data

    expected = np.mean(
        np.stack([np.full((4, 4), i + 1, dtype=np.float32) for i in range(3)], axis=0),
        axis=0,
    )
    assert np.allclose(data, expected)
    os.remove(out)

