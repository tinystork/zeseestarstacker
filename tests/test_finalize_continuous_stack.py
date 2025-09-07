import numpy as np
from astropy.io import fits


def test_finalize_continuous_stack_crops(tmp_path, monkeypatch):
    import types
    import sys
    from pathlib import Path

    if "seestar.gui" not in sys.modules:
        root = Path(__file__).resolve().parents[1]
        seestar_pkg = types.ModuleType("seestar")
        seestar_pkg.__path__ = [str(root / "seestar")]
        gui_pkg = types.ModuleType("seestar.gui")
        gui_pkg.__path__ = [str(root / "seestar" / "gui")]
        settings_mod = types.ModuleType("seestar.gui.settings")
        settings_mod.SettingsManager = object
        gui_pkg.settings = settings_mod
        seestar_pkg.gui = gui_pkg
        sys.modules["seestar"] = seestar_pkg
        sys.modules["seestar.gui"] = gui_pkg
        sys.modules["seestar.gui.settings"] = settings_mod
        zmod = types.ModuleType("zemosaic")
        zmod.zemosaic_config = types.SimpleNamespace(
            get_astap_default_search_radius=lambda: 0
        )
        sys.modules.setdefault("zemosaic", zmod)

    from seestar.queuep.queue_manager import SeestarQueuedStacker

    obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    wht = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    sum_ = np.zeros((4, 4, 3), dtype=np.float32)
    sum_[1:3, 1:3, :] = 10.0

    obj.cumulative_wht_memmap = wht
    obj.cumulative_sum_memmap = sum_
    hdr = fits.Header()
    hdr["CRPIX1"] = 1.0
    hdr["CRPIX2"] = 1.0
    hdr["NAXIS1"] = 4
    hdr["NAXIS2"] = 4
    obj.reference_header_for_wcs = hdr

    monkeypatch.chdir(tmp_path)
    obj.finalize_continuous_stack()
    with fits.open(tmp_path / "master_stack_classic_nodriz.fits") as hdul:
        data = hdul[0].data
        hdr_out = hdul[0].header

    assert data.shape == (2, 2, 3)
    assert hdr_out["CRPIX1"] == 0.0
    assert hdr_out["CRPIX2"] == 0.0
    assert np.all(data == 10.0)

