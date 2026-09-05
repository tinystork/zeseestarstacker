"""RF-2 — production-seam tests: the real registration alignment contract.

These exercise the **real** production functions (``SeestarAligner._align_image``
and ``SeestarQueuedStacker`` seams), with only the heavy solver/I/O
monkeypatched, to evidence the worker/reference lifecycle rather than infer it
from RF-1 prose:

* ``_align_image`` discards the astroalign similarity scale (``alignment.py:228-237``)
  and returns the 2x3 ``return_M`` matrix used by the Drizzle standard path.
* ``_align_image(return_M=True)`` returns ``M=None`` on every failure path.
* the resume contract: ``_is_plain_classic`` is False for reproject/drizzle/mosaic
  (only plain classic SUM/W is resumable), and resume artifacts fail closed.
* ``_solve_cumulative_stack`` keeps the WCS grid frozen under ``freeze_reference_wcs``
  (the frozen-grid / evolving-data distinction).

No production file under ``seestar/`` is modified.
"""

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RF2_RESEARCH = ROOT / "research" / "registration_reference_rf2"
sys.path.insert(0, str(RF2_RESEARCH))

# --- import the real alignment module (imports cleanly, no GUI stubs needed) ---
from seestar.core import alignment as alignment_mod  # noqa: E402
from seestar.core.alignment import SeestarAligner  # noqa: E402
from skimage.transform import SimilarityTransform  # noqa: E402

import m16_target_policy_witness as tpw  # noqa: E402


def _make_star_image(shape=(64, 64)):
    img = np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for cx, cy in ((32, 20), (20, 40), (44, 44)):
        img += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.8 ** 2)))
    return (img / img.max()).astype(np.float32)


def _known_transform(scale=1.02, rotation_deg=2.0, tx=5.0, ty=-3.0):
    t = np.radians(rotation_deg)
    params = np.array(
        [
            [scale * np.cos(t), -scale * np.sin(t), tx],
            [scale * np.sin(t), scale * np.cos(t), ty],
            [0.0, 0.0, 1.0],
        ]
    )
    return SimilarityTransform(matrix=params), params


# --------------------------------------------------------------------------
# real _align_image: scale-discard + return_M contract
# --------------------------------------------------------------------------


def test_align_image_discards_scale_and_returns_M(monkeypatch):
    aligner = SeestarAligner()
    T, params = _known_transform()

    def fake_find_transform(source, target):
        return T, (np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 0.0], [1.0, 1.0]]))

    monkeypatch.setattr(alignment_mod.aa, "find_transform", fake_find_transform)

    aligned, success, M = aligner._align_image(
        _make_star_image(), _make_star_image(), "s.fits", return_M=True
    )

    assert success is True
    assert M is not None
    assert M.shape == (2, 3), "return_M must be the 2x3 affine actually used by warpAffine"
    assert M.dtype == np.float64
    # scale discarded: the linear 2x2 has unit determinant (forced scale = 1.0)
    assert abs(np.linalg.det(M[:2, :2]) - 1.0) < 1e-9
    # rotation and translation preserved exactly (theta recovered from atan2)
    assert abs(np.degrees(np.arctan2(M[1, 0], M[0, 0])) - 2.0) < 1e-9
    assert abs(M[0, 2] - 5.0) < 1e-9
    assert abs(M[1, 2] - (-3.0)) < 1e-9
    # the aligned image is a real float32 warp of the input (shape preserved)
    assert aligned.shape == (64, 64)
    assert aligned.dtype == np.float32


def test_align_image_failure_returns_M_none(monkeypatch):
    aligner = SeestarAligner()

    def fake_find_transform_none(source, target):
        return None, (None, None)

    monkeypatch.setattr(alignment_mod.aa, "find_transform", fake_find_transform_none)

    aligned, success, M = aligner._align_image(
        _make_star_image(), _make_star_image(), "s.fits", return_M=True
    )
    assert success is False
    assert M is None, "M must be None on the failure path (no matrix was computed)"


# --------------------------------------------------------------------------
# resume contract + frozen-grid semantics (queue_manager seams)
# --------------------------------------------------------------------------


def _import_qm():
    """Import queue_manager with GUI stubs (same pattern as the RF-1/HSI tests)."""
    saved = {n: sys.modules.get(n) for n in ("seestar", "seestar.gui",
                                              "seestar.gui.settings",
                                              "seestar.gui.histogram_widget")}
    if "seestar.gui" not in sys.modules:
        seestar_pkg = types.ModuleType("seestar")
        seestar_pkg.__path__ = [str(ROOT / "seestar")]
        gui_pkg = types.ModuleType("seestar.gui")
        gui_pkg.__path__ = []
        settings_mod = types.ModuleType("seestar.gui.settings")
        settings_mod.SettingsManager = type("DummySettingsManager", (), {})
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

    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod
    return qm


def _make_wcs(shape=(4, 4)):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.01, 0.01])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


def test_is_plain_classic_false_for_reproject_drizzle_mosaic():
    qm = _import_qm()
    obj = qm.SeestarQueuedStacker()

    obj.is_mosaic_run = False
    obj.drizzle_active_session = False
    obj.reproject_between_batches = False
    obj.reproject_coadd_final = False
    assert obj._is_plain_classic() is True, "plain classic SUM/W must be resumable"

    for kwargs in (
        {"drizzle_active_session": True},
        {"reproject_between_batches": True},
        {"reproject_coadd_final": True},
        {"is_mosaic_run": True},
    ):
        obj.is_mosaic_run = False
        obj.drizzle_active_session = False
        obj.reproject_between_batches = False
        obj.reproject_coadd_final = False
        for k, v in kwargs.items():
            setattr(obj, k, v)
        assert obj._is_plain_classic() is False, (
            f"{kwargs} must not be treated as the resumable plain classic mode"
        )


def test_resume_artifacts_present_detects_manifest(tmp_path):
    qm = _import_qm()
    obj = qm.SeestarQueuedStacker()
    obj.output_folder = str(tmp_path)
    # no artifacts -> not present
    assert obj._resume_artifacts_present(str(tmp_path)) is False
    # a manifest (or any resume artifact) -> present (fail-closed signal)
    (tmp_path / "memmap_accumulators").mkdir()
    (tmp_path / "memmap_accumulators" / qm._RESUME_MANIFEST_FILENAME).write_text("{}")
    assert obj._resume_artifacts_present(str(tmp_path)) is True


def test_solve_cumulative_stack_keeps_wcs_frozen(monkeypatch, tmp_path):
    qm = _import_qm()
    from astropy.io import fits

    obj = qm.SeestarQueuedStacker()
    obj.update_progress = lambda *a, **k: None
    obj.freeze_reference_wcs = True
    obj.reproject_between_batches = True
    obj.memmap_shape = (4, 4, 3)
    obj.cumulative_sum_memmap = np.ones(obj.memmap_shape, dtype=np.float32)
    obj.cumulative_wht_memmap = np.ones(obj.memmap_shape, dtype=np.float32)
    wcs_initial = _make_wcs(shape=(4, 4))
    obj.reference_header_for_wcs = wcs_initial.to_header()
    obj.ref_wcs_header = obj.reference_header_for_wcs.copy()
    obj.reference_wcs_object = wcs_initial

    solver_called = {"n": 0}

    def fake_solver(path):
        solver_called["n"] += 1
        return True

    monkeypatch.setattr(qm.SeestarQueuedStacker, "_run_solver_and_update_header", fake_solver)

    stack, hdr = obj._solve_cumulative_stack()

    # freeze_reference_wcs -> solver skipped, grid stays frozen
    assert solver_called["n"] == 0
    assert stack is not None
    assert np.allclose(obj.reference_wcs_object.wcs.crval, [0, 0])


# --------------------------------------------------------------------------
# real M16 target-policy witness (corrective C1) — actual prepared pixels +
# real astroalign matcher + production Euclidean conversion
# --------------------------------------------------------------------------


def test_target_policy_production_euclidean_discards_scale():
    T, _ = _known_transform(scale=1.02, rotation_deg=2.0, tx=5.0, ty=-3.0)
    M = tpw.production_euclidean(T)
    assert M.shape == (3, 3)
    assert abs(np.linalg.det(M[:2, :2]) - 1.0) < 1e-9, "scale must be discarded"
    assert abs(np.degrees(np.arctan2(M[1, 0], M[0, 0])) - 2.0) < 1e-9
    assert abs(M[0, 2] - 5.0) < 1e-9
    assert abs(M[1, 2] - (-3.0)) < 1e-9


def test_target_policy_apply_handles_single_and_batch():
    M = np.eye(3)
    out = tpw._apply(M, np.array([10.0, 20.0]))
    assert out.shape == (1, 2)
    assert np.allclose(out[0], [10.0, 20.0])
    out2 = tpw._apply(M, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert out2.shape == (2, 2)
    assert np.allclose(out2, [[1.0, 2.0], [3.0, 4.0]])


def test_target_policy_dispersion_zero_when_identical():
    M = SimilarityTransform(rotation=np.radians(1.0), translation=(3.0, -2.0)).params
    canonical = {"centre": np.array([50.0, 50.0]), "corner": np.array([99.0, 99.0])}
    configs = {f"c{i}": {j: M for j in range(5)} for i in range(3)}
    d = tpw._dispersion(configs, canonical)
    for pt in ("centre", "corner"):
        assert d[pt]["max"] == 0.0


def test_target_policy_dispersion_detects_difference():
    M0 = SimilarityTransform(rotation=0.0, translation=(0.0, 0.0)).params
    M1 = SimilarityTransform(rotation=0.0, translation=(1.0, 0.0)).params
    canonical = {"corner": np.array([0.0, 0.0])}
    configs = {"a": {0: M0}, "b": {0: M1}}
    d = tpw._dispersion(configs, canonical)
    assert abs(d["corner"]["max"] - 1.0) < 1e-9


@pytest.mark.skipif(not os.path.isdir(tpw.M16_FOLDER), reason="M16 dataset not present")
def test_target_policy_witness_integration():
    # bounded subset: 6 frames, 2 batch sizes, 2 orders — exercises the full
    # immutable + evolving code path without the full-matrix runtime.
    r = tpw.run(
        tpw.M16_FOLDER,
        seed=0,
        batch_sizes=(1, 10),
        orders=("natural", "reversed"),
        max_frames=6,
    )
    assert r["n_others"] == 6
    # immutable target: exact organization invariance on real pixels
    for pt in ("centre", "edge", "corner"):
        assert r["dispersion"]["immutable_batch"][pt]["max"] == 0.0
        assert r["dispersion"]["immutable_order"][pt]["max"] == 0.0
    # every frame aligned against the immutable reference
    assert r["immutable"]["natural"]["n_failed"] == 0
