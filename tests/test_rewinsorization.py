import numpy as np
from zemosaic.zemosaic_align_stack import _reject_outliers_winsorized_sigma_clip


def test_rewinsor_replace_values():
    stack = np.stack([
        np.zeros((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32) * 10,
        np.ones((2, 2), dtype=np.float32) * 10,
        np.ones((2, 2), dtype=np.float32) * 10,
    ], axis=0)
    out, _ = _reject_outliers_winsorized_sigma_clip(
        stack, (0.25, 0.25), 1.0, 1.0, apply_rewinsor=True
    )
    assert not np.isnan(out).any()
    assert np.allclose(out[0], 10.0)


def test_rewinsor_nan_when_disabled():
    stack = np.stack([
        np.zeros((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32) * 10,
        np.ones((2, 2), dtype=np.float32) * 10,
        np.ones((2, 2), dtype=np.float32) * 10,
    ], axis=0)
    out, _ = _reject_outliers_winsorized_sigma_clip(
        stack, (0.25, 0.25), 1.0, 1.0, apply_rewinsor=False
    )
    assert np.isnan(out[0]).all()


def test_stack_batch_calls_winsor(monkeypatch, tmp_path):
    import importlib
    import sys
    import types
    from pathlib import Path
    from astropy.io import fits

    ROOT = Path(__file__).resolve().parents[1]
    if "seestar.gui" not in sys.modules:
        seestar_pkg = types.ModuleType("seestar")
        seestar_pkg.__path__ = [str(ROOT / "seestar")]
        gui_pkg = types.ModuleType("seestar.gui")
        gui_pkg.__path__ = []
        settings_mod = types.ModuleType("seestar.gui.settings")

        class DummySettingsManager:
            pass

        settings_mod.SettingsManager = DummySettingsManager
        hist_mod = types.ModuleType("seestar.gui.histogram_widget")
        hist_mod.HistogramWidget = object
        gui_pkg.settings = settings_mod
        gui_pkg.histogram_widget = hist_mod
        seestar_pkg.gui = gui_pkg
        sys.modules["seestar"] = seestar_pkg
        sys.modules["seestar.gui"] = gui_pkg
        sys.modules["seestar.gui.settings"] = settings_mod
        sys.modules["seestar.gui.histogram_widget"] = hist_mod

    qm = importlib.import_module("seestar.queuep.queue_manager")

    calls = {"n": 0}

    def fake_winsor(self, images, weights, kappa=3.0, winsor_limits=(0.05, 0.05), apply_rewinsor=True, **kwargs):
        calls["n"] += 1
        return np.zeros_like(images[0]), 0.0

    monkeypatch.setattr(qm.SeestarQueuedStacker, "_stack_winsorized_sigma", fake_winsor)

    obj = qm.SeestarQueuedStacker()
    obj.update_progress = lambda *a, **k: None
    obj.stacking_mode = "winsorized-sigma"
    obj.stack_kappa_low = 2.5
    obj.stack_kappa_high = 2.5
    obj.winsor_limits = (0.05, 0.05)
    obj.use_quality_weighting = False
    obj.reproject_between_batches = False

    img = np.ones((2, 2), dtype=np.float32)
    hdr = fits.Header()
    item = (img, hdr, {}, None, np.ones((2, 2), dtype=bool))

    out, hdr_out, cov = obj._stack_batch([item], current_batch_num=1, total_batches_est=1)

    assert calls["n"] == 1
    assert hdr_out["STK_NOTE"] == "Stacked with winsorized sigma clip"
