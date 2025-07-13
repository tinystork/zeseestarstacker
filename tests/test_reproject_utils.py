import importlib.util
import sys
from pathlib import Path

import pytest

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


def test_channel_orientation_chw(monkeypatch):
    module = reproject_utils

    def dummy_reproj(data_wcs, output_projection=None, shape_out=None, **kwargs):
        data, _ = data_wcs
        return data[: shape_out[0], : shape_out[1]], np.ones(shape_out, dtype=float)

    from astropy.wcs import WCS
    import numpy as np

    wcs = WCS(naxis=2)
    wcs.pixel_shape = (1, 1)

    chw_data = np.ones((3, 1, 1), dtype=np.float32)
    result, cov = module.reproject_and_coadd(
        [(chw_data, wcs)],
        output_projection=wcs,
        shape_out=(1, 1),
        reproject_function=dummy_reproj,
    )

    assert result.shape == (1, 1, 3)
    assert np.allclose(result, 1)
    assert np.allclose(cov, 1)



def test_channel_orientation_with_weights(monkeypatch):
    module = reproject_utils

    def dummy_reproj(data_wcs, output_projection=None, shape_out=None, **kwargs):
        data, _ = data_wcs
        return data[: shape_out[0], : shape_out[1]], np.ones(shape_out, dtype=float)

    from astropy.wcs import WCS
    import numpy as np

    wcs = WCS(naxis=2)
    wcs.pixel_shape = (1, 1)

    chw_data = np.full((3, 1, 1), 2.0, dtype=np.float32)
    chw_weight = np.ones((3, 1, 1), dtype=np.float32)
    result, cov = module.reproject_and_coadd(
        [(chw_data, wcs)],
        output_projection=wcs,
        shape_out=(1, 1),
        input_weights=[chw_weight],
        reproject_function=dummy_reproj,
    )

    assert result.shape == (1, 1, 3)
    assert np.allclose(result, 2)
    assert np.allclose(cov, 1)


def test_zero_coverage_fallback(monkeypatch):
    module = reproject_utils

    def dummy_reproj(data_wcs, output_projection=None, shape_out=None, **kwargs):
        data, _ = data_wcs
        return data[: shape_out[0], : shape_out[1]], np.ones(shape_out, dtype=float)

    from astropy.wcs import WCS
    import numpy as np

    wcs = WCS(naxis=2)
    wcs.pixel_shape = (1, 1)

    data = np.ones((1, 1), dtype=np.float32)
    result, cov = module.reproject_and_coadd(
        [(data, wcs)],
        output_projection=wcs,
        shape_out=(1, 1),
        input_weights=[np.zeros((1, 1), dtype=np.float32)],
        reproject_function=dummy_reproj,
    )

    assert np.allclose(result, 1)
    assert np.allclose(cov, 1)

