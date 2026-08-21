"""Reliability R2 — Drizzle ``drizzle_scale`` must expand the output grid.

Mission ZSSS-QT-RELIABILITY-AUDIT, anomaly B: the human witness produced a real
``stack_final_drizzle_final`` at (near) native resolution — a Drizzle ×2 request
yielded ~1081×1921 @ 2.37 arcsec/px instead of ~2160×3840 @ 1.185 arcsec/px.

Root cause (engine regression): commit ``3bc34a8f`` ("Fix freeze_reference_wcs
grid prep") added an ``elif self.freeze_reference_wcs and (...)`` branch in
``SeestarQueuedStacker.start_processing`` that — after ``_prepare_global_reprojection_grid()``
had already produced the *native* reference grid — **overwrote**
``self.drizzle_output_wcs`` / ``self.drizzle_output_shape_hw`` with the native
reference grid, discarding the ×2 grid that ``_create_drizzle_output_wcs`` had
correctly produced earlier in the same method (``scale_factor=self.drizzle_scale``).

``freeze_reference_wcs`` is auto-enabled by ``batch_size == 0`` (the Qt default),
so the default Qt run path hits the overwrite.

These tests lock the contract:
    Drizzle Standard ×N  ->  output grid ~N× reference (both axes) and
                             pixel scale ~ reference_scale / N;
    no drizzle           ->  output grid stays at the reference (native) grid.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub seestar.gui.settings so queue_manager imports without Tk (same pattern as
# test_m3d_policy.py / test_reliability_source_immutability_r1.py).
if "seestar.gui.settings" not in sys.modules:
    settings_mod = types.ModuleType("seestar.gui.settings")

    class DummySettingsManager:
        pass

    settings_mod.SettingsManager = DummySettingsManager
    settings_mod.TILE_HEIGHT = 512
    sys.modules["seestar.gui.settings"] = settings_mod
    hist_mod = types.ModuleType("seestar.gui.histogram_widget")
    hist_mod.HistogramWidget = object
    sys.modules["seestar.gui.histogram_widget"] = hist_mod

qm = importlib.import_module("seestar.queuep.queue_manager")

from seestar.gui.run_config import build_backend_kwargs, split_backend_kwargs  # noqa: E402
from seestar.gui_qt.settings_state import QtSettingsState  # noqa: E402

H = 64
W = 64
PIXSCALE_ARCSEC = 2.37


def make_native_wcs(h=H, w=W, scale_arcsec=PIXSCALE_ARCSEC):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    wcs.wcs.crval = [274.7499, -13.8200]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    cdelt = scale_arcsec / 3600.0
    wcs.wcs.cdelt = [-cdelt, cdelt]
    wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
    wcs.pixel_shape = (w, h)
    wcs.array_shape = (h, w)
    return wcs


def wcs_pixscale_arcsec(wcs_obj):
    scales = proj_plane_pixel_scales(wcs_obj)
    vals = []
    for s in scales:
        try:
            vals.append(float(s.to_value("deg")))
        except Exception:
            vals.append(float(s))
    return float(np.mean(np.abs(vals))) * 3600.0


def _make_stackers(tmp_path):
    """Full-init stacker with stubbed aligner/solver (deterministic native WCS)."""
    stacker = qm.SeestarQueuedStacker()
    stacker.set_progress_callback(lambda *a, **k: None)

    class DummyAligner:
        def __init__(self):
            self.correct_hot_pixels = True
            self.hot_pixel_threshold = 3.0
            self.neighborhood_size = 5
            self.bayer_pattern = "GRBG"

        def _get_reference_image(self, folder, files, out_folder):
            data = np.zeros((H, W, 3), dtype=np.float32)
            hdr = make_native_wcs().to_header()
            hdr.insert(0, ("NAXIS", 2))
            hdr.insert(1, ("NAXIS1", W))
            hdr.insert(2, ("NAXIS2", H))
            ref_path = Path(out_folder) / "temp_processing" / "reference_image.fit"
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            fits.writeto(
                ref_path,
                np.zeros((H, W), dtype=np.float32),
                header=hdr,
                overwrite=True,
            )
            return data, hdr

    stacker.aligner = DummyAligner()
    stacker._solve_astrometry_async = lambda *a, **k: make_native_wcs()

    # One real FITS in the input dir (so _add_files_to_queue has something).
    src = Path(tmp_path) / "Light_001.fit"
    fits.writeto(
        src,
        np.zeros((H, W), dtype=np.float32),
        header=make_native_wcs().to_header(),
        overwrite=True,
    )
    return stacker


def _run(tmp_path, use_drizzle, scale):
    """Drive ``start_processing`` and return the resulting grid state."""
    in_dir = Path(tmp_path) / "in"
    out_dir = Path(tmp_path) / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    # copy fresh FITS into the input folder
    src = Path(tmp_path) / "Light_001.fit"
    fits.writeto(
        in_dir / "Light_001.fit",
        np.zeros((H, W), dtype=np.float32),
        header=make_native_wcs().to_header(),
        overwrite=True,
    )
    del src

    state = QtSettingsState(
        input_folder=str(in_dir),
        output_folder=str(out_dir),
        temp_folder=str(out_dir),
        output_filename="stack_final",
        use_drizzle=use_drizzle,
        drizzle_scale=scale,
        drizzle_wht_threshold=0.7,
        drizzle_mode="Final",
        drizzle_kernel="square",
        drizzle_pixfrac=1.0,
        drizzle_group_size=50,
        batch_size=0,  # Qt default -> freeze_reference_wcs auto-enabled
    )
    start_kwargs, _ = split_backend_kwargs(build_backend_kwargs(state))

    stacker = _make_stackers(in_dir)
    try:
        started = stacker.start_processing(**start_kwargs)
    finally:
        stacker.stop()
        try:
            stacker.processing_thread.join(timeout=3)
        except Exception:
            pass
    assert started is True
    return stacker


# ---------------------------------------------------------------------------
# 1. handoff e2e: Qt state -> backend kwargs carries drizzle_scale as float
# ---------------------------------------------------------------------------


def test_handoff_drizzle_scale_reaches_backend_kwargs():
    state = QtSettingsState(use_drizzle=True, drizzle_scale=2)
    kw = build_backend_kwargs(state)
    assert kw["drizzle_scale"] == 2.0
    assert kw["use_drizzle"] is True
    assert kw["drizzle_mode"] == "Final"
    assert kw["drizzle_wht_threshold"] == 0.7
    assert kw["drizzle_kernel"] == "square"
    assert kw["drizzle_pixfrac"] == 1.0
    assert kw["drizzle_group_size"] == 50


# ---------------------------------------------------------------------------
# 2. scale effect: Drizzle Standard ×2 must expand the grid and halve the scale
# ---------------------------------------------------------------------------


def test_drizzle_scale_x2_expands_output_grid(tmp_path):
    stacker = _run(tmp_path, use_drizzle=True, scale=2)

    assert stacker.drizzle_scale == 2.0
    assert stacker.drizzle_active_session is True
    shape_hw = stacker.drizzle_output_shape_hw
    assert shape_hw == (H * 2, W * 2), (
        f"drizzle output grid must be 2x reference, got {shape_hw}"
    )
    # accumulator shape must match the output grid (no mismatch)
    acc = stacker.drizzle_accumulators[0]
    assert acc.wht.shape == (H * 2, W * 2)
    # output WCS pixel scale must be reference / 2 = 1.185 arcsec/px
    out_scale = wcs_pixscale_arcsec(stacker.drizzle_output_wcs)
    assert abs(out_scale - PIXSCALE_ARCSEC / 2.0) < 0.02, (
        f"output pixel scale must be ~1.185 arcsec/px, got {out_scale}"
    )


# ---------------------------------------------------------------------------
# 3. ×3 (wide tolerance)
# ---------------------------------------------------------------------------


def test_drizzle_scale_x3_expands_output_grid(tmp_path):
    stacker = _run(tmp_path, use_drizzle=True, scale=3)

    shape_hw = stacker.drizzle_output_shape_hw
    assert shape_hw == (H * 3, W * 3), f"expected 3x grid, got {shape_hw}"
    out_scale = wcs_pixscale_arcsec(stacker.drizzle_output_wcs)
    assert abs(out_scale - PIXSCALE_ARCSEC / 3.0) < 0.03, (
        f"output pixel scale must be ~0.79 arcsec/px, got {out_scale}"
    )


# ---------------------------------------------------------------------------
# 4. counter-test: no drizzle -> grid stays at reference (native) resolution
# ---------------------------------------------------------------------------


def test_no_drizzle_grid_stays_native(tmp_path):
    stacker = _run(tmp_path, use_drizzle=False, scale=2)

    assert stacker.drizzle_active_session is False
    shape_hw = stacker.drizzle_output_shape_hw
    # With drizzle OFF no up-sampled drizzle output grid is created; it must
    # never be the ×2 grid (None or the native reference grid are both fine).
    assert shape_hw in (None, (H, W)), (
        f"no drizzle -> no expanded grid, got {shape_hw}"
    )
