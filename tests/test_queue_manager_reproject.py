import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reproject_utils",
    ROOT / "seestar" / "enhancement" / "reproject_utils.py",
)
reproject_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproject_utils)


def make_wcs(shape=(10, 10)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.01, 0.01])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


class DummyStacker:
    def __init__(self):
        self.memmap_shape = (10, 10, 3)
        self.reference_wcs_object = make_wcs(shape=self.memmap_shape[:2])
        self.cumulative_sum_memmap = np.zeros(self.memmap_shape, dtype=np.float32)
        self.cumulative_wht_memmap = np.zeros(self.memmap_shape[:2], dtype=np.float32)
        self.reproject_between_batches = False

    # copy of implemented method
    def _reproject_to_reference(self, image_array, input_wcs):
        reproject_interp = reproject_utils.reproject_interp
        target_shape = self.memmap_shape[:2]
        if image_array.ndim == 3:
            channels = []
            footprint = None
            for ch in range(image_array.shape[2]):
                reproj_ch, footprint = reproject_interp(
                    (image_array[..., ch], input_wcs),
                    self.reference_wcs_object,
                    shape_out=target_shape,
                )
                channels.append(reproj_ch)
            result = np.stack(channels, axis=2)
        else:
            result, footprint = reproject_interp(
                (image_array, input_wcs),
                self.reference_wcs_object,
                shape_out=target_shape,
            )
        return result.astype(np.float32), footprint.astype(np.float32)

    def _combine_batch_result(self, data, header, coverage, batch_wcs=None):
        batch_sum = data.astype(np.float32)
        batch_wht = coverage.astype(np.float32)
        if not self.reproject_between_batches and self.reference_wcs_object and batch_wcs is not None:
            reproject_interp = reproject_utils.reproject_interp
            shp = self.memmap_shape[:2]
            if batch_sum.ndim == 3:
                channels = []
                for ch in range(batch_sum.shape[2]):
                    c, _ = reproject_interp((batch_sum[..., ch], batch_wcs), self.reference_wcs_object, shape_out=shp)
                    channels.append(c)
                batch_sum = np.stack(channels, axis=2)
            else:
                batch_sum, _ = reproject_interp((batch_sum, batch_wcs), self.reference_wcs_object, shape_out=shp)
            batch_wht, _ = reproject_interp((batch_wht, batch_wcs), self.reference_wcs_object, shape_out=shp)
        self.cumulative_sum_memmap += batch_sum
        self.cumulative_wht_memmap += batch_wht


def test_reproject_to_reference_rgb():
    s = DummyStacker()
    img = np.ones((5, 5, 3), dtype=np.float32)
    wcs_in = make_wcs(shape=(5, 5))
    out, foot = s._reproject_to_reference(img, wcs_in)
    assert out.shape == s.memmap_shape
    assert foot.shape == s.memmap_shape[:2]


def test_combine_batch_respects_flag(monkeypatch):
    s = DummyStacker()
    img = np.ones(s.memmap_shape, dtype=np.float32)
    cov = np.ones(s.memmap_shape[:2], dtype=np.float32)
    wcs_in = make_wcs(shape=s.memmap_shape[:2])

    calls = {"n": 0}

    def fake_reproj(*args, **kwargs):
        calls["n"] += 1
        return args[0][0], np.ones(s.memmap_shape[:2], dtype=np.float32)

    monkeypatch.setattr(reproject_utils, "reproject_interp", fake_reproj)
    s.reproject_between_batches = False
    s._combine_batch_result(img, fits.Header(), cov, wcs_in)
    assert calls["n"] > 0

    calls["n"] = 0
    s.cumulative_sum_memmap.fill(0)
    s.cumulative_wht_memmap.fill(0)
    s.reproject_between_batches = True
    s._combine_batch_result(img, fits.Header(), cov, wcs_in)
    assert calls["n"] == 0
