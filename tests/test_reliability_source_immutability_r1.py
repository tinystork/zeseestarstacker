"""Reliability R1 — source-folder immutability when ``move_stacked=False``.

Mission ZSSS-QT-RELIABILITY-AUDIT, anomaly A: the human witness observed a run
where files were announced "Moved to stacked: ..." and later runs failed with
"File not found: .../quick/Light_....fit", ending in "Drizzle: aucune image
accumulée".

Root cause (engine regression, pre-Qt): ``SeestarQueuedStacker._move_to_stacked``
moved processed RAW files into a ``stacked/`` sibling folder *unconditionally*,
even though ``move_stacked`` defaults to ``False``.  Commit ``3b47a3e`` (and the
later batch-plan commit ``1cf6450``) introduced unguarded
``self._move_to_stacked(...)`` calls that bypass the flag; the original
``0358f88`` design guarded every move with ``if self.move_stacked``.

These tests lock the invariant:
    ``move_stacked=False`` (explicit safety mode) -> NO source file is ever moved;
    ``move_stacked=True``  (historical default)    -> source moved to ``stacked/``
        only after successful consumption.
"""

import importlib
import queue
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

# Stub GUI modules to avoid Tk dependence during import (same pattern as
# test_m3d_policy.py / test_worker_incremental_drizzle.py).  The stubs are
# restored right after the queue-manager import so a full ``pytest tests/``
# collection never leaks a fake ``seestar.gui`` (empty ``__path__``) into
# sibling engine tests that import ``seestar.gui.run_config`` etc.
_saved_sys_modules = {
    name: sys.modules.get(name)
    for name in ("seestar", "seestar.gui", "seestar.gui.settings", "seestar.gui.histogram_widget")
}
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

# Restore the real packages so this module's import never pollutes the rest
# of the test session (see comment above).
for _name, _mod in _saved_sys_modules.items():
    if _mod is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _mod


def make_wcs(shape=(2, 2)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.01, 0.01])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


# --------------------------------------------------------------------------
# A) targeted choke-point tests (_move_to_stacked respects move_stacked)
# --------------------------------------------------------------------------


def _lightweight_obj(move_stacked):
    """Minimal instance for the _move_to_stacked unit test (no worker pools)."""
    obj = qm.SeestarQueuedStacker.__new__(qm.SeestarQueuedStacker)
    obj.move_stacked = bool(move_stacked)
    obj.stacked_subdir_name = "stacked"
    obj.update_progress = lambda *a, **k: None
    return obj


def _write_fits(tmp_path, name="Light_001.fit", shape=(8, 8)):
    p = Path(tmp_path) / name
    fits.writeto(p, np.zeros(shape, dtype=np.float32), overwrite=True)
    return p


def test_move_stacked_false_does_not_move_sources(tmp_path):
    p = _write_fits(tmp_path)
    obj = _lightweight_obj(move_stacked=False)
    obj._move_to_stacked([str(p)])

    assert p.exists(), "source file must stay in place when move_stacked=False"
    assert not (Path(tmp_path) / "stacked").exists(), (
        "no 'stacked' subdir may be created when move_stacked=False"
    )


def test_move_stacked_true_moves_sources(tmp_path):
    p = _write_fits(tmp_path)
    obj = _lightweight_obj(move_stacked=True)
    obj._move_to_stacked([str(p)])

    assert not p.exists(), "source file must be moved when move_stacked=True"
    moved = Path(tmp_path) / "stacked" / p.name
    assert moved.exists(), "moved file must land in <src>/stacked/<base>"


def test_move_stacked_false_ignores_missing_paths(tmp_path):
    obj = _lightweight_obj(move_stacked=False)
    obj._move_to_stacked([str(tmp_path / "does_not_exist.fit")])  # no raise


# --------------------------------------------------------------------------
# B) minimal real worker run (drizzle standard) — source stays intact
# --------------------------------------------------------------------------


def _make_minimal_worker(tmp_path, move_stacked=None):
    """Full-init stacker driven through one drizzle-standard worker iteration.

    Mirrors ``tests/test_worker_incremental_drizzle.py`` but deliberately does
    NOT stub ``_move_to_stacked`` so the real move gate is exercised.
    """
    obj = qm.SeestarQueuedStacker()
    obj.perform_cleanup = False
    obj.stop_processing = False
    obj.current_folder = str(tmp_path)
    obj.output_folder = str(tmp_path)
    obj.queue = queue.Queue()
    src = Path(tmp_path) / "Light_001.fit"
    fits.writeto(
        src, np.zeros((2, 2, 3), dtype=np.float32), overwrite=True
    )
    obj.queue.put(str(src))
    obj.additional_folders = []
    obj.files_in_queue = 1
    obj.batch_size = 1
    obj.drizzle_active_session = True
    obj.drizzle_mode = "Final"  # M3: drizzle_mode is now without effect
    obj.stacked_batches_count = 0
    obj.total_batches_estimated = 1
    obj.mosaic_settings_dict = {}
    obj.local_solver_preference = "none"
    obj.astap_search_radius = 1.0
    obj.astap_downsample = 1
    obj.astap_sensitivity = 100
    obj.reference_pixel_scale_arcsec = 1.0
    obj.astap_path = ""
    obj.astap_data_dir = ""
    obj.local_ansvr_path = ""
    obj.api_key = None
    obj.ansvr_timeout_sec = 5
    obj.astap_timeout_sec = 5
    obj.astrometry_net_timeout_sec = 5
    obj.drizzle_fillval = "0.0"
    obj.update_progress = lambda *a, **k: None
    # Closure-B restore: the constructor default is now True (historical
    # filesystem checkpoint).  Zero-mutation tests must opt out explicitly.
    if move_stacked is not None:
        obj.move_stacked = bool(move_stacked)

    # Stub simple reference FITS for _get_reference_image
    ref_path = Path(tmp_path) / "temp_processing" / "reference_image.fit"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(ref_path, np.zeros((2, 2), dtype=np.float32), overwrite=True)

    class DummyAligner:
        def __init__(self):
            self.correct_hot_pixels = True
            self.hot_pixel_threshold = 3.0
            self.neighborhood_size = 5
            self.bayer_pattern = "GRBG"

        def _get_reference_image(self, folder, files, out_folder):
            return np.zeros((2, 2, 3), dtype=np.float32), fits.Header()

    obj.aligner = DummyAligner()

    class DummySolver:
        def solve(self, *a, **k):
            return make_wcs()

    obj.astrometry_solver = DummySolver()
    obj._create_drizzle_output_wcs = lambda ref_wcs, shape, scale: (
        make_wcs(shape),
        shape,
    )

    dummy_data = np.zeros((2, 2, 3), dtype=np.float32)
    tf = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]], dtype=np.float64)
    obj._process_file = lambda *a, **k: (
        dummy_data,
        fits.Header(),
        None,
        None,
        tf,
        np.ones((2, 2), dtype=np.float32),
    )

    calls = {"add_frame": 0}

    def fake_add_frame(original_data, header, tf_val, weight_map, native_wcs=None):
        calls["add_frame"] += 1
        obj.stop_processing = True
        return True

    obj._add_frame_to_drizzle_accumulators = fake_add_frame
    # NOTE: _move_to_stacked is intentionally NOT stubbed.
    obj._save_partial_stack = lambda *a, **k: None
    obj._update_batch_count_file = lambda *a, **k: None
    obj._send_eta_update = lambda *a, **k: None
    obj._save_final_stack = lambda *a, **k: None
    obj._process_incremental_drizzle_batch = (
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("incremental called"))
    )
    obj._start_drizzle_process = (
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("incremental start called"))
    )
    obj._process_and_save_drizzle_batch = (
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("final batch called"))
    )
    obj._process_completed_batch = (
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("classic called"))
    )
    obj.cleanup_temp_reference = lambda: None
    obj._cleanup_drizzle_temp_files = lambda: None
    obj._cleanup_drizzle_batch_outputs = lambda: None
    obj._cleanup_mosaic_panel_stacks_temp = lambda: None
    obj._wait_drizzle_processes = lambda: None

    return obj, calls


def test_worker_minimal_run_keeps_source_when_move_stacked_false(tmp_path):
    src = Path(tmp_path) / "Light_001.fit"
    obj, calls = _make_minimal_worker(tmp_path, move_stacked=False)
    qm.SeestarQueuedStacker._worker(obj)

    assert calls["add_frame"] == 1
    assert src.exists(), (
        "after a minimal drizzle run with move_stacked=False the source RAW "
        "must remain in place"
    )
    assert not (Path(tmp_path) / "stacked").exists(), (
        "no 'stacked' subdir may be created when move_stacked=False"
    )


def test_constructor_default_move_stacked_is_true():
    obj = qm.SeestarQueuedStacker()
    try:
        assert obj.move_stacked is True, (
            "closure-B restore: the constructor default must be True "
            "(historical filesystem checkpoint)"
        )
    finally:
        # Avoid leaving background pools alive beyond the test.
        for attr in ("quality_executor",):
            ex = getattr(obj, attr, None)
            if ex is not None:
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass


def test_worker_minimal_run_moves_source_by_default(tmp_path):
    """Mirror of the immutability test: with the restored default
    (move_stacked left unset -> True), a successfully consumed source RAW
    must be moved into <src>/stacked/.
    """
    src = Path(tmp_path) / "Light_001.fit"
    obj, calls = _make_minimal_worker(tmp_path)  # constructor default (True)
    assert obj.move_stacked is True
    qm.SeestarQueuedStacker._worker(obj)

    assert calls["add_frame"] == 1
    assert not src.exists(), (
        "with the default move_stacked=True the consumed source RAW must be moved"
    )
    moved = Path(tmp_path) / "stacked" / src.name
    assert moved.exists(), "moved file must land in <src>/stacked/<base>"
