import importlib.util
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "stack_methods", ROOT / "seestar" / "core" / "stack_methods.py"
)
stack_methods = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stack_methods)
_stack_winsorized_sigma = stack_methods._stack_winsorized_sigma


def _generate_data(color=False):
    rng = np.random.default_rng(0)
    N, H, W = 20, 32, 32
    if color:
        base = rng.normal(100.0, 5.0, size=(N, H, W, 3)).astype(np.float32)
        clean = base.copy()
        mask = rng.random((N, H, W, 1)) < 0.05
        base[mask.repeat(3, axis=-1)] = rng.uniform(150, 200, size=np.count_nonzero(mask) * 3)
    else:
        base = rng.normal(100.0, 5.0, size=(N, H, W)).astype(np.float32)
        clean = base.copy()
        mask = rng.random((N, H, W)) < 0.05
        base[mask] = rng.uniform(150, 200, size=mask.sum())
    return base, clean


def test_winsorized_sigma_monochrome():
    data, clean = _generate_data(False)
    result, pct = _stack_winsorized_sigma(data, None, kappa=3.0, winsor_limits=(0.05, 0.05))
    assert 3 < pct < 10
    mean_true = np.mean(clean)
    assert np.isclose(np.mean(result), mean_true, rtol=0.02)


def test_winsorized_sigma_color_with_weights():
    data, clean = _generate_data(True)
    rng = np.random.default_rng(1)
    weights = rng.random(data.shape[0]).astype(np.float32)
    result, pct = _stack_winsorized_sigma(data, weights, kappa=3.0, winsor_limits=(0.05, 0.05))
    assert 3 < pct < 10
    w = weights[:, None, None, None]
    expected = np.sum(clean * w, axis=0) / np.sum(w, axis=0)
    mean_true = np.mean(expected)
    assert np.isclose(np.mean(result), mean_true, rtol=0.02)
