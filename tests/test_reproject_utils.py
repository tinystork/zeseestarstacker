import importlib.util
import sys
from pathlib import Path

import pytest

import types
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "reproject_utils",
    ROOT / "seestar" / "enhancement" / "reproject_utils.py",
)
reproject_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproject_utils)
reproject_and_coadd = reproject_utils.reproject_and_coadd
reproject_interp = reproject_utils.reproject_interp


def test_functions_callable():
    assert callable(reproject_and_coadd)
    assert callable(reproject_interp)


def test_missing_reproject(monkeypatch):
    module_name = "reproject_utils_missing"
    # reload module under a different name after patching
    monkeypatch.setitem(sys.modules, "reproject", None)
    monkeypatch.setitem(sys.modules, "reproject.mosaicking", None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "seestar" / "enhancement" / "reproject_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ImportError) as exc:
        module.reproject_and_coadd([], None, (1, 1))
    assert "pip install reproject" in str(exc.value)


def test_fallback_on_wcs_mismatch(monkeypatch):
    module = reproject_utils

    def raise_value_error(*args, **kwargs):
        raise ValueError("The two WCS return a different number of world coordinates")

    monkeypatch.setattr(module, "_astropy_reproject_and_coadd", raise_value_error)

    def dummy_reproj(data_wcs, output_projection=None, shape_out=None, **kwargs):
        data, _ = data_wcs
        return data[:shape_out[0], :shape_out[1]], np.ones(shape_out, dtype=float)

    from astropy.wcs import WCS
    import numpy as np

    wcs = WCS(naxis=2)
    wcs.pixel_shape = (1, 1)
    result, cov = module.reproject_and_coadd(
        [(np.ones((1, 1), dtype=np.float32), wcs)],
        output_projection=wcs,
        shape_out=(1, 1),
        reproject_function=dummy_reproj,
    )
    assert np.allclose(result, 1)
    assert np.allclose(cov, 1)


def test_fallback_on_generic_error(monkeypatch):
    module = reproject_utils

    def raise_type_error(*args, **kwargs):
        raise TypeError("Output shape mismatch")

    monkeypatch.setattr(module, "_astropy_reproject_and_coadd", raise_type_error)

    def dummy_reproj(data_wcs, output_projection=None, shape_out=None, **kwargs):
        data, _ = data_wcs
        return data[: shape_out[0], : shape_out[1]], np.ones(shape_out, dtype=float)

    from astropy.wcs import WCS
    import numpy as np

    wcs = WCS(naxis=2)
    wcs.pixel_shape = (1, 1)
    result, cov = module.reproject_and_coadd(
        [(np.ones((1, 1), dtype=np.float32), wcs)],
        output_projection=wcs,
        shape_out=(1, 1),
        reproject_function=dummy_reproj,
    )
    assert np.allclose(result, 1)
    assert np.allclose(cov, 1)


def test_skip_non_celestial_inputs(monkeypatch):
    module = reproject_utils

    def dummy_reproj(data_wcs, output_projection=None, shape_out=None, **kwargs):
        data, _ = data_wcs
        return data[: shape_out[0], : shape_out[1]], np.ones(shape_out, dtype=float)


    def fake_astropy_coadd(input_data, **kwargs):
        # ensure only celestial WCS inputs are forwarded
        assert len(input_data) == 1
        _, wcs_obj = input_data[0]
        assert getattr(wcs_obj, "has_celestial", False)
        shape_out = kwargs.get("shape_out")
        return np.ones(shape_out, dtype=float), np.ones(shape_out, dtype=float)

    monkeypatch.setattr(module, "_astropy_reproject_and_coadd", fake_astropy_coadd)


    from astropy.wcs import WCS
    import numpy as np

    wcs_cel = WCS(naxis=2)
    wcs_cel.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_cel.pixel_shape = (1, 1)

    wcs_non = WCS(naxis=2)
    wcs_non.wcs.ctype = ["LON", "LAT"]
    wcs_non.pixel_shape = (1, 1)

    result, cov = module.reproject_and_coadd(
        [(np.ones((1, 1), dtype=np.float32), wcs_non), (np.ones((1, 1), dtype=np.float32), wcs_cel)],
        output_projection=wcs_cel,
        shape_out=(1, 1),
        reproject_function=dummy_reproj,
    )

    assert np.allclose(result, 1)
    assert np.allclose(cov, 1)


def test_memory_threshold_forces_fallback(monkeypatch):
    module = reproject_utils

    monkeypatch.setenv("REPROJECT_MEM_THRESHOLD_GB", "0")
    called = {"n": 0}

    def fake_astropy(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("should not be called")

    monkeypatch.setattr(module, "_astropy_reproject_and_coadd", fake_astropy)

    def dummy_reproj(data_wcs, output_projection=None, shape_out=None, **kwargs):
        data, _ = data_wcs
        return data[: shape_out[0], : shape_out[1]], np.ones(shape_out, dtype=float)

    from astropy.wcs import WCS
    import numpy as np

    wcs = WCS(naxis=2)
    wcs.pixel_shape = (1, 1)

    result, cov = module.reproject_and_coadd(
        [(np.ones((1, 1), dtype=np.float32), wcs)],
        output_projection=wcs,
        shape_out=(1, 1),
        reproject_function=dummy_reproj,
    )

    assert np.allclose(result, 1)
    assert np.allclose(cov, 1)
    assert called["n"] == 0


def test_no_duplicate_return_footprint(monkeypatch):
    module = reproject_utils

    def fake_astropy(input_data, output_projection=None, shape_out=None, **kwargs):
        reproj_fn = kwargs["reproject_function"]
        return reproj_fn(
            input_data[0],
            output_projection=output_projection,
            shape_out=shape_out,
            return_footprint=True,
        )

    monkeypatch.setattr(module, "_astropy_reproject_and_coadd", fake_astropy)

    def dummy_reproj(data_wcs, output_projection=None, shape_out=None, return_footprint=False, **kwargs):
        data, _ = data_wcs
        assert return_footprint is True
        return data[: shape_out[0], : shape_out[1]], np.ones(shape_out, dtype=float)

    from astropy.wcs import WCS
    import numpy as np

    wcs = WCS(naxis=2)
    wcs.pixel_shape = (1, 1)

    result, cov = module.reproject_and_coadd(
        [(np.ones((1, 1), dtype=np.float32), wcs)],
        output_projection=wcs,
        shape_out=(1, 1),
        reproject_function=dummy_reproj,
    )

    assert np.allclose(result, 1)
    assert np.allclose(cov, 1)


def test_streaming_accepts_mixed_channels(tmp_path):
    module = reproject_utils
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np

    hdr = WCS(naxis=2).to_header()
    mono = tmp_path / "mono.fits"
    fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32), hdr).writeto(mono)
    rgb = tmp_path / "rgb.fits"
    fits.PrimaryHDU(np.ones((3, 4, 4), dtype=np.float32), hdr).writeto(rgb)

    def dummy_reproj(data_wcs, output_projection=None, shape_out=None, return_footprint=True, **kwargs):
        data, _ = data_wcs
        arr = np.asarray(data, dtype=np.float32)
        footprint = np.ones_like(arr, dtype=np.float32)
        return arr, footprint

    out = tmp_path / "out.fits"
    ok = module.streaming_reproject_and_coadd(
        [str(mono), str(rgb)],
        output_path=str(out),
        reproject_function=dummy_reproj,
    )
    assert ok
    with fits.open(out) as hdul:
        assert hdul[0].data.shape == (3, 4, 4)


def test_streaming_raises_when_no_contribution(tmp_path):
    module = reproject_utils
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np

    hdr = WCS(naxis=2).to_header()
    mono = tmp_path / "mono.fits"
    fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32), hdr).writeto(mono)

    def failing_reproj(*args, **kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="Aucune contribution reçue"):
        module.streaming_reproject_and_coadd(
            [str(mono)],
            output_path=str(tmp_path / "out.fits"),
            reproject_function=failing_reproj,
        )
