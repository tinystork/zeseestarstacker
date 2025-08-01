import importlib.util
from pathlib import Path
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

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
        self.reproject_between_batches = True
        self.images_in_cumulative_stack = 0

    def _reproject_batch_to_reference(self, img, wht, wcs_in):
        reproject_interp = reproject_utils.reproject_interp
        shp = self.memmap_shape[:2]
        if img.ndim == 3:
            chs = []
            for c in range(img.shape[2]):
                d, _ = reproject_interp((img[:, :, c], wcs_in), self.reference_wcs_object, shape_out=shp)
                chs.append(d)
            img_out = np.stack(chs, axis=2)
        else:
            img_out, _ = reproject_interp((img, wcs_in), self.reference_wcs_object, shape_out=shp)
        wht_out, _ = reproject_interp((wht, wcs_in), self.reference_wcs_object, shape_out=shp)
        return img_out.astype(np.float32), wht_out.astype(np.float32)

    def _combine_batch_result(self, data, header, coverage):
        signal = data.astype(np.float64) * coverage.astype(np.float64)[:, :, np.newaxis]
        self.cumulative_sum_memmap += signal.astype(np.float32)
        self.cumulative_wht_memmap += coverage.astype(np.float32)
        self.images_in_cumulative_stack += 1


class DummyStackerWithHeaderCount(DummyStacker):
    def _combine_batch_result(self, data, header, coverage):
        count = int(header.get("NIMAGES", 1))
        signal = data.astype(np.float64) * coverage.astype(np.float64)[:, :, np.newaxis]
        self.cumulative_sum_memmap += signal.astype(np.float32)
        self.cumulative_wht_memmap += coverage.astype(np.float32)
        self.images_in_cumulative_stack += count


def test_inter_batch_reprojection_range(tmp_path):
    wcs_in = make_wcs()
    data = np.random.random((10, 10)).astype(np.float32)
    paths = []
    for i in range(3):
        p = tmp_path / f"im{i}.fits"
        fits.writeto(p, data, header=wcs_in.to_header(), overwrite=True)
        paths.append(p)

    s = DummyStacker()

    for p in paths:
        img = fits.getdata(p).astype(np.float32)
        hdr = fits.getheader(p)
        img_rgb = np.stack([img] * 3, axis=2)
        img_reproj, cov_reproj = s._reproject_batch_to_reference(img_rgb, np.ones((10, 10), dtype=np.float32), wcs_in)
        s._combine_batch_result(img_reproj, hdr, cov_reproj)

    result = s.cumulative_sum_memmap / s.cumulative_wht_memmap[:, :, np.newaxis]
    assert result.min() >= 0
    assert result.max() <= 1


def test_cumulative_stack_counter():
    s = DummyStacker()

    for _ in range(4):
        img = np.ones(s.memmap_shape, dtype=np.float32)
        cov = np.ones(s.memmap_shape[:2], dtype=np.float32)
        s._combine_batch_result(img, fits.Header(), cov)

    assert s.images_in_cumulative_stack == 4


def test_images_in_cumulative_stack_header_count():
    s = DummyStackerWithHeaderCount()

    hdr1 = fits.Header()
    hdr1["NIMAGES"] = 2
    hdr2 = fits.Header()
    hdr2["NIMAGES"] = 3

    img = np.ones(s.memmap_shape, dtype=np.float32)
    cov = np.ones(s.memmap_shape[:2], dtype=np.float32)

    s._combine_batch_result(img, hdr1, cov)
    s._combine_batch_result(img, hdr2, cov)

    assert s.images_in_cumulative_stack == 5
