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
