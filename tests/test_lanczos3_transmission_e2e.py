"""A1/A5 end-to-end Lanczos kernel transmission closure (real wiring).

The reported Windows discrepancy was "user intended Lanczos3 x3, durable log
showed kernel=lanczos2 scale=3.0".  These tests pin the FULL production chain
so any future value-changing seam is caught:

* Qt combo ``currentText`` -> ``QtSettingsState.drizzle_kernel``
  (``collect_settings_state``),
* ``QtSettingsState`` -> ``RunRequest.backend_kwargs`` (canonical builder),
* ``backend_kwargs`` -> ``start_processing`` surface via
  ``split_backend_kwargs`` (the exact seam ``SeestarQueuedStackerBackend``
  uses),
* the real engine kernel-validation boundary (``validate_drizzle_kernel`` ->
  effective kernel on the instance and on the real accumulators) plus the
  durable ``DRIZZLE_CONFIG`` log line emitted by ``initialize``,
* the real ``run_config.cfg`` writer (``DrizzleCheckpointWriter`` commit ->
  canonical config serialisation), read back and asserted.

The engine value is NEVER mocked: tests 3-4 drive the real
``SeestarQueuedStacker`` / ``DrizzleCheckpointWriter`` production code with the
same kernel value the Qt request carried.  Lanczos2 is exercised
independently of Lanczos3, exactly like the mission requires.

No display, no stacking of real FITS, no long engine runs.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from seestar.core.drizzle_checkpoint import (
    DrizzleCheckpointWriter,
    build_drizzle_canonical_config,
)
from seestar.core.drizzle_core import DrizzleAccumulator, validate_drizzle_kernel

SHAPE = (8, 8)


def _reference_geometry(shape=SHAPE):
    """A minimal valid v2 frozen input-reference geometry payload."""
    from astropy.wcs import WCS
    from seestar.core.drizzle_checkpoint import serialize_input_reference_geometry

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [shape[1] / 2.0 + 0.5, shape[0] / 2.0 + 0.5]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.array_shape = shape
    return serialize_input_reference_geometry(wcs, shape, None)
_ENGINE_SHAPE = (32, 32)  # proven reference grid for real initialize()


def _wcs(shape=SHAPE):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2.0 + 0.5, shape[0] / 2.0 + 0.5]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-0.001, 0.001]
    w.array_shape = shape
    return w


# --------------------------------------------------------------------------
# 1. Qt combo -> state -> RunRequest -> start_processing surface (real Qt)
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    from seestar.gui_qt import create_application

    app = create_application([])
    assert app is QApplication.instance()
    return app


@pytest.fixture()
def window(qapp):
    from seestar.gui_qt import MainWindow

    win = MainWindow()
    yield win
    win.shutdown()


@pytest.mark.parametrize("kernel", ["lanczos3", "lanczos2"])
def test_qt_combo_selection_reaches_request_and_start_surface(window, kernel):
    """Combo -> state -> request/backend kwargs -> start kwargs (real Qt)."""
    from seestar.gui_qt.run_bridge import split_backend_kwargs

    window.drizzle_check.setChecked(True)
    window.drizzle_kernel_combo.setCurrentText(kernel)
    window.drizzle_scale_spin.setValue(3)
    window.drizzle_pixfrac_spin.setValue(0.6)
    window.drizzle_wht_spin.setValue(0.3)

    # 1. combo text -> settings state.
    state = window.collect_settings_state()
    assert state.drizzle_kernel == kernel
    assert state.drizzle_scale == 3

    # 2. state -> canonical RunRequest backend kwargs.
    request = window.build_run_request()
    assert request.backend_kwargs["drizzle_kernel"] == kernel
    assert request.backend_kwargs["drizzle_scale"] == 3.0
    assert request.backend_kwargs["drizzle_pixfrac"] == pytest.approx(0.6)
    assert request.backend_kwargs["drizzle_wht_threshold"] == pytest.approx(0.3)

    # 3. backend kwargs -> the exact start_processing surface the
    #    SeestarQueuedStackerBackend adapter builds (no index remap, text kept).
    start_kwargs, seam_kwargs = split_backend_kwargs(request.backend_kwargs)
    assert "drizzle_kernel" not in seam_kwargs
    assert start_kwargs["drizzle_kernel"] == kernel
    assert start_kwargs["drizzle_scale"] == 3.0


def test_qt_combo_restore_is_text_based_not_index_based(window):
    """Selection uses exact kernel text; removing an item can never shift a
    user's Lanczos3 selection onto Lanczos2 (live state sync included)."""
    window.drizzle_kernel_combo.setCurrentText("lanczos3")
    assert window.drizzle_kernel_combo.currentText() == "lanczos3"
    # currentText -> currentIndex round trip preserves the exact kernel name.
    idx = window.drizzle_kernel_combo.currentIndex()
    assert window.drizzle_kernel_combo.itemText(idx) == "lanczos3"
    # The combo change signal live-syncs the settings state (no stale value).
    assert window.settings_state.drizzle_kernel == "lanczos3"
    assert window.collect_settings_state().drizzle_kernel == "lanczos3"


# --------------------------------------------------------------------------
# 2. Real engine validation boundary + durable DRIZZLE_CONFIG line
# --------------------------------------------------------------------------


def _real_qm_initialized(tmp_path, kernel, *, scale, pixfrac, threshold):
    """Real SeestarQueuedStacker + real initialize() with an active standard
    (non-mosaic) drizzle session configured from the request values."""
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    qm = SeestarQueuedStacker()
    qm.drizzle_active_session = True
    qm.is_mosaic_run = False
    qm.reproject_between_batches = False
    qm.reproject_coadd_final = False
    qm.reference_wcs_object = _wcs(_ENGINE_SHAPE)
    qm.drizzle_scale = scale
    qm.drizzle_kernel = str(kernel)
    qm.drizzle_pixfrac = pixfrac
    qm.drizzle_wht_threshold = threshold
    qm.drizzle_output_wcs = _wcs(_ENGINE_SHAPE)
    qm.drizzle_output_shape_hw = _ENGINE_SHAPE
    progress = []

    def recorder(message, _progress=None, _level=None):
        progress.append(str(message))

    qm.update_progress = recorder
    assert (
        qm.initialize(str(tmp_path), (_ENGINE_SHAPE[0], _ENGINE_SHAPE[1], 3)) is True
    )
    return qm, progress


@pytest.mark.parametrize("kernel", ["lanczos3", "lanczos2"])
def test_real_engine_effective_kernel_and_durable_config_line(
    tmp_path, kernel
):
    """Requested lanczos kernel -> engine effective kernel + durable line."""
    qm, progress = _real_qm_initialized(
        tmp_path, kernel, scale=3.0, pixfrac=0.6, threshold=0.3
    )

    # Engine effective kernel == the requested kernel (validate passes it
    # through; nothing remaps lanczos3 -> lanczos2).
    eff, reason = validate_drizzle_kernel(kernel)
    assert eff == kernel and reason is None
    assert qm.drizzle_kernel == kernel
    for acc in qm.drizzle_accumulators:
        assert acc.kernel == kernel

    # The durable DRIZZLE_CONFIG line records BOTH effective and requested.
    lines = [m for m in progress if m.startswith("DRIZZLE_CONFIG")]
    assert lines, "no DRIZZLE_CONFIG durable line emitted by initialize"
    tokens = lines[0].split()
    assert f"kernel={kernel}" in tokens
    assert f"requested_kernel={kernel}" in tokens
    assert "scale=3.0" in tokens
    assert "requested_scale=3.0" in tokens
    # Lanczos policy: effective pixfrac/threshold forced, requested recorded.
    assert "pixfrac=1.0" in tokens
    assert "requested_pixfrac=0.6" in tokens
    assert "wht_threshold=0.0" in tokens
    assert "requested_wht_threshold=0.3" in tokens
    # The effective token is exactly the selected kernel (no lanczos3->2 remap).
    other = "lanczos2" if kernel == "lanczos3" else "lanczos3"
    assert f"kernel={other}" not in tokens


@pytest.mark.parametrize("kernel", ["lanczos3", "lanczos2"])
def test_engine_validate_never_remaps_lanczos3_to_lanczos2(kernel):
    eff, reason = validate_drizzle_kernel(kernel)
    assert eff == kernel
    assert reason is None


# --------------------------------------------------------------------------
# 3. run_config.cfg durable record (real writer, real serialisation)
# --------------------------------------------------------------------------


def _seed_run_config(tmp_path, kernel, *, scale=3.0):
    """Seed a real drizzle checkpoint generation via the real writer so the
    canonical run_config.cfg lands on disk (production first-commit path)."""
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir(parents=True)
    inputs.mkdir(parents=True)
    paths = []
    for i in range(4):
        p = inputs / f"src_{i}.fit"
        hdu = fits.PrimaryHDU(np.full(SHAPE, i + 1, dtype=np.uint16))
        hdu.header["EXPTIME"] = 1.0
        hdu.writeto(p)
        paths.append(p)

    def identity(path):
        st = os.stat(path)
        return {
            "path": os.path.normcase(str(path)),
            "name": os.path.basename(str(path)),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }

    idents = [identity(p) for p in paths]

    qm = object.__new__(SeestarQueuedStacker)
    qm.output_folder = str(output)
    qm.drizzle_active_session = True
    qm.is_mosaic_run = False
    qm.reproject_between_batches = False
    qm.reproject_coadd_final = False
    qm.move_stacked = False
    qm.stacked_subdir_name = "stacked"
    qm.weighting_method = "none"
    qm.use_quality_weighting = False
    qm.weight_by_snr = True
    qm.weight_by_stars = True
    qm.snr_exponent = 1.0
    qm.stars_exponent = 0.5
    qm.min_weight = 0.01
    qm.correct_hot_pixels = True
    qm.hot_pixel_threshold = 3.0
    qm.neighborhood_size = 5
    qm.bayer_pattern = "GRBG"
    qm.drizzle_scale = scale
    qm.drizzle_kernel = kernel
    qm.drizzle_pixfrac = 1.0
    qm.drizzle_wht_threshold = 0.0
    qm.drizzle_wht_threshold_effective = 0.0
    qm.drizzle_fillval = "0.0"
    qm.drizzle_group_size = 2
    qm.drizzle_processing_policy = "incremental"

    cfg = build_drizzle_canonical_config(
        qm, product_version=qm._canonical_product_version()
    )
    writer = DrizzleCheckpointWriter(
        output, qm._canonical_product_version(), cfg, _wcs(), SHAPE
    )
    accs = [DrizzleAccumulator(SHAPE, kernel=kernel, pixfrac=1.0) for _ in range(3)]
    yy, xx = np.indices(SHAPE, dtype=np.float64)
    for i in range(2):
        data = (i * 3.0 + xx * 0.25 + yy * 0.5).astype(np.float32)
        weight = np.full(SHAPE, 0.7 + i * 0.05, dtype=np.float32)
        pixmap = np.dstack((xx + i * 0.07, yy - i * 0.04))
        for acc in accs:
            acc.add(
                data,
                weight,
                pixmap,
                exptime=1.0,
                in_units="counts",
                in_grid_mask=np.ones(SHAPE, dtype=bool),
            )
    writer.commit(
        accs,
        session_binding={
            "input_roots": [str(inputs)],
            "reference": idents[0],
            "plan": {
                "sources": idents,
                "decomposition": [4],
            },
            "reference_geometry": _reference_geometry(),
        },
        counters={
            "frame_count": 2,
            "stacked_batches_count": 2,
            "total_exposure_seconds": 2.0,
            "exposure_unknown_count": 0,
            "exposure_min": 1.0,
            "exposure_max": 1.0,
        },
        completed_sources=idents[:2],
    )
    return output / "run_config.cfg"


@pytest.mark.parametrize("kernel", ["lanczos3", "lanczos2"])
def test_run_config_cfg_records_requested_kernel(tmp_path, kernel):
    """The durable run_config.cfg records kernel=<lanczosX> (real writer)."""
    import seestar.run_contract as run_contract

    cfg_path = _seed_run_config(tmp_path, kernel, scale=3.0)
    assert cfg_path.is_file()

    report = run_contract.read_cfg(str(cfg_path))
    assert report.config.scientific["drizzle_kernel_effective"] == kernel
    assert (
        report.config.scientific["drizzle_scale_effective"]
        == pytest.approx(3.0)
    )


def test_run_config_cfg_distinguishes_lanczos2_from_lanczos3(tmp_path):
    """The cfg writer records each kernel faithfully (no cross-wiring)."""
    import seestar.run_contract as run_contract

    cfg3 = _seed_run_config(tmp_path, "lanczos3", scale=3.0)
    cfg2_path = tmp_path / "second" / "run_config.cfg"
    cfg2_path.parent.mkdir(parents=True)
    import shutil

    _seed_run_config(tmp_path / "seed2", "lanczos2", scale=3.0)
    shutil.copy(tmp_path / "seed2" / "out" / "run_config.cfg", cfg2_path)

    r3 = run_contract.read_cfg(str(cfg3)).config.scientific
    r2 = run_contract.read_cfg(str(cfg2_path)).config.scientific
    assert r3["drizzle_kernel_effective"] == "lanczos3"
    assert r2["drizzle_kernel_effective"] == "lanczos2"
