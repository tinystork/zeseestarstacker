"""RF2 — production implementation tests: immutable target + passive diagnostics
+ Drizzle transform-only (skip-resampling) contract.

These exercise the *real* production functions (``SeestarAligner._align_image``
and ``SeestarQueuedStacker`` seams / ``_process_file``) with only the heavy
matcher/solver/I/O monkeypatched:

* ``_align_image`` transform-only returns the same 2x3 tf as the regular path for
  a deterministic matcher, and does **not** invoke the warp backend.
* classic ``_align_image`` still warps exactly once.
* failure return contracts (2-/3-/4-tuple) remain correct.
* passive diagnostics do not alter ``M``/output and fail open on I/O errors.
* ``_process_file`` Drizzle returns the original prepared data / mask / tf and
  does not call the warp backend (dead pre-warp removed).
* the registration target is never reassigned to a stack (structural pin).
"""

import importlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from skimage.transform import SimilarityTransform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seestar.core import alignment as alignment_mod  # noqa: E402
from seestar.core.alignment import SeestarAligner  # noqa: E402
from seestar.core.registration_diagnostics import (  # noqa: E402
    SCHEMA_VERSION,
    TARGET_POLICY,
    append_record,
    build_record,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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


def _fake_find_transform(T):
    def fake(source, target):
        pts = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 40.0], [50.0, 10.0]])
        tgt = T(pts)
        return T, (pts, tgt)

    return fake


# ---------------------------------------------------------------------------
# transform-only contract
# ---------------------------------------------------------------------------


def test_transform_only_tf_equals_regular_tf(monkeypatch):
    aligner = SeestarAligner()
    T, _ = _known_transform()
    monkeypatch.setattr(alignment_mod.aa, "find_transform", _fake_find_transform(T))

    src = _make_star_image()
    ref = _make_star_image()

    aligned, ok, M = aligner._align_image(src, ref, "s.fits", return_M=True)
    orig, ok2, M2 = aligner._align_image(
        src, ref, "s.fits", return_M=True, transform_only=True
    )

    assert ok is True and ok2 is True
    assert M is not None and M2 is not None
    assert M.shape == (2, 3) and M2.shape == (2, 3)
    # the tf is identical (same matcher + same Euclidean scale-discard)
    np.testing.assert_array_equal(M, M2)
    # regular path returned a warped float32 image ...
    assert aligned.shape == (64, 64)
    assert aligned.dtype == np.float32
    # ... transform-only returned the ORIGINAL image (no resampling)
    assert orig is src or np.array_equal(orig, src)


def test_transform_only_does_not_call_warp_backend(monkeypatch):
    aligner = SeestarAligner()
    T, _ = _known_transform()
    monkeypatch.setattr(alignment_mod.aa, "find_transform", _fake_find_transform(T))

    cpu_calls = []

    def spy_cpu(self, img, M, dsize, out=None):
        cpu_calls.append(1)
        raise AssertionError("warp backend must not be called in transform-only mode")

    monkeypatch.setattr(SeestarAligner, "_align_cpu", spy_cpu)

    src = _make_star_image()
    ref = _make_star_image()
    _, ok, M = aligner._align_image(
        src, ref, "s.fits", return_M=True, transform_only=True
    )
    assert ok is True
    assert M is not None
    assert cpu_calls == []


def test_classic_still_warps_exactly_once(monkeypatch):
    import cv2 as _cv2

    aligner = SeestarAligner()
    T, _ = _known_transform(scale=1.0, rotation_deg=0.0, tx=3.0, ty=-2.0)
    monkeypatch.setattr(alignment_mod.aa, "find_transform", _fake_find_transform(T))

    real_warp = _cv2.warpAffine
    calls = []

    def spy_warp(src, M, dsize, *a, **k):
        calls.append(1)
        return real_warp(src, M, dsize, *a, **k)

    monkeypatch.setattr(alignment_mod.cv2, "warpAffine", spy_warp)

    src = _make_star_image()  # 2D -> one warpAffine call
    ref = _make_star_image()
    aligned, ok = aligner._align_image(src, ref, "s.fits")

    assert ok is True
    assert len(calls) == 1, "classic alignment must warp exactly once"


def test_failure_return_contracts(monkeypatch):
    aligner = SeestarAligner()

    def fail(source, target):
        return None, (None, None)

    monkeypatch.setattr(alignment_mod.aa, "find_transform", fail)
    src = _make_star_image()
    ref = _make_star_image()

    # 2-tuple (default)
    r2 = aligner._align_image(src, ref, "s.fits")
    assert r2 == (src, False)

    # 3-tuple (return_M)
    img, ok, M = aligner._align_image(src, ref, "s.fits", return_M=True)
    assert ok is False and M is None and img is src

    # 4-tuple (return_diagnostics)
    img, ok, M, diag = aligner._align_image(src, ref, "s.fits", return_diagnostics=True)
    assert ok is False and M is None and diag is None

    # 4-tuple (transform_only + return_diagnostics)
    img, ok, M, diag = aligner._align_image(
        src, ref, "s.fits", transform_only=True, return_diagnostics=True
    )
    assert ok is False and M is None and diag is None


# ---------------------------------------------------------------------------
# passive diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_do_not_alter_M_or_output(monkeypatch):
    aligner = SeestarAligner()
    T, _ = _known_transform()
    monkeypatch.setattr(alignment_mod.aa, "find_transform", _fake_find_transform(T))

    src = _make_star_image()
    ref = _make_star_image()

    a_img, a_ok, a_M = aligner._align_image(src, ref, "s.fits", return_M=True)
    b_img, b_ok, b_M, diag = aligner._align_image(
        src, ref, "s.fits", return_M=True, return_diagnostics=True
    )

    assert a_ok is b_ok is True
    np.testing.assert_array_equal(a_M, b_M)
    np.testing.assert_array_equal(a_img, b_img)

    # diagnostic content is truthful and diagnostic-only
    assert diag is not None
    assert diag["model"] == "euclidean"
    assert abs(diag["raw_scale"] - 1.02) < 1e-6
    assert abs(diag["applied_rotation_deg"] - 2.0) < 1e-6
    assert abs(diag["applied_translation"][0] - 5.0) < 1e-6
    assert abs(diag["applied_translation"][1] - (-3.0)) < 1e-6
    assert diag["match_count"] == 4
    # residual under the APPLIED (scale-discarded) matrix: scale 1.02 discarded
    # -> non-zero residual even though the matches are exact under the similarity
    assert diag["residual_px"] is not None
    assert diag["residual_px"]["rms"] > 0.0


def test_build_record_schema():
    rec = build_record(
        frame="sub_0001.fit",
        reference_provenance="ref.fit",
        success=True,
        raw_scale=1.001,
        applied_rotation_deg=0.01,
        applied_translation=[1.0, -2.0],
        match_count=10,
        residual_px={"p50": 0.1, "p95": 0.2, "rms": 0.11},
        session_id="s1",
    )
    assert rec["schema_version"] == SCHEMA_VERSION
    assert rec["target_policy"] == TARGET_POLICY
    assert rec["event"] == "registration"
    assert rec["frame"] == "sub_0001.fit"
    assert rec["reference_provenance"] == "ref.fit"
    assert "raw_scale" in rec["diagnostic_only"]
    assert "residual_px" in rec["diagnostic_only"]


def test_append_record_and_fail_open(tmp_path, monkeypatch):
    path = tmp_path / "out" / "registration_diagnostics.jsonl"
    ok = append_record(str(path), build_record(frame="a.fit", success=True))
    assert ok is True
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["frame"] == "a.fit"

    # fail-open: a path whose parent is a *file* cannot be written -> returns
    # False and never raises
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    bad = blocker / "registration_diagnostics.jsonl"
    assert append_record(str(bad), build_record(frame="b.fit", success=True)) is False

    # also fail-open when json serialization cannot proceed (non-serializable)
    assert append_record(
        str(path), build_record(frame="c.fit", success=True, raw_scale=object())
    ) is False


def test_qm_diagnostics_fail_open(tmp_path, monkeypatch):
    """_record_registration_diagnostics must never raise, even with a broken
    output folder / writer."""
    qm = _import_qm()
    obj = object.__new__(qm.SeestarQueuedStacker)
    obj.output_folder = str(tmp_path / "out")
    obj._registration_session_id = "s1"
    obj._registration_target_provenance_id = "ref.fit"

    # normal write
    obj._record_registration_diagnostics("f.fit", True, {"model": "euclidean"})
    path = Path(obj.output_folder) / "registration_diagnostics.jsonl"
    assert path.exists()
    rec = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert rec["frame"] == "f.fit"
    assert rec["reference_provenance"] == "ref.fit"
    assert rec["target_policy"] == TARGET_POLICY

    # broken writer (append_record raises) -> swallowed, no exception
    def boom(path_, record_):
        raise OSError("disk full")

    monkeypatch.setattr(
        "seestar.core.registration_diagnostics.append_record", boom
    )
    obj._record_registration_diagnostics("g.fit", True, None)  # must not raise

    # no output folder -> no-op, no exception
    obj.output_folder = None
    obj._record_registration_diagnostics("h.fit", True, None)


# ---------------------------------------------------------------------------
# _process_file Drizzle: dead pre-warp removed
# ---------------------------------------------------------------------------


def _import_qm():
    saved = {
        n: sys.modules.get(n)
        for n in (
            "seestar",
            "seestar.gui",
            "seestar.gui.settings",
            "seestar.gui.histogram_widget",
        )
    }
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


def _make_wcs(shape=(32, 32)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.001, 0.001])
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


def _drizzle_qm(shape):
    from seestar.core.drizzle_core import DrizzleAccumulator, build_output_grid

    qm = _import_qm()
    obj = object.__new__(qm.SeestarQueuedStacker)
    obj.align_on_disk = False
    obj.bayer_pattern = "GRBG"
    obj.correct_hot_pixels = False
    obj.drizzle_active_session = True
    obj.is_mosaic_run = False
    obj.use_quality_weighting = False
    obj.batch_size = 0
    obj.reproject_between_batches = False
    obj.reproject_coadd_final = False
    obj.reference_wcs_object = _make_wcs(shape)
    obj.update_progress = lambda *a, **k: None
    obj.warned_unaligned_source_folders = set()
    obj.aligner = SeestarAligner()
    obj.aligner.update_progress = lambda *a, **k: None
    obj.drizzle_output_wcs, out_shape = build_output_grid(
        obj.reference_wcs_object, shape, 1.0
    )
    obj.drizzle_accumulators = [DrizzleAccumulator(out_shape) for _ in range(3)]
    return qm, obj


def _gauss(shape, amp, sig, pos):
    h, w = shape
    yy, xx = np.indices((h, w))
    return (
        amp * np.exp(-((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2) / (2.0 * sig ** 2))
    ).astype(np.float32)


def test_process_file_drizzle_returns_original_data_mask_tf(tmp_path, monkeypatch):
    shape = (64, 64)
    qm_mod, obj = _drizzle_qm(shape)

    # deterministic matcher: exact translation (+4,+6)
    exact = SimilarityTransform(rotation=0.0, translation=(4.0, 6.0))
    monkeypatch.setattr(
        alignment_mod.aa,
        "find_transform",
        lambda source, target: (exact, ([], [])),
    )

    # build an RGB frame with a bright star at (20,20) and a dark block
    img = np.full((shape[0], shape[1], 3), 100.0, dtype=np.float32)
    img[10:15, 10:15, :] = 0.0
    g = _gauss(shape, 400.0, 2.0, (20.0, 20.0))
    for c in range(3):
        img[..., c] += g
    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = shape[1]
    hdr["NAXIS2"] = shape[0]
    hdr["NAXIS3"] = 3
    path = tmp_path / "in.fits"
    fits.PrimaryHDU(data=img, header=hdr).writeto(path, overwrite=True)
    ref_data = np.zeros((shape[0], shape[1], 3), dtype=np.float32)

    # spy the warp backend to prove it is NOT called on the Drizzle path
    cpu_calls = []
    orig_cpu = SeestarAligner._align_cpu

    def spy_cpu(self, im, M, dsize, out=None):
        cpu_calls.append(1)
        return orig_cpu(self, im, M, dsize, out=out)

    monkeypatch.setattr(SeestarAligner, "_align_cpu", spy_cpu)

    data, header, _scores, _wcs, matrix_m, mask = obj._process_file(
        str(path), ref_data, solve_astrometry_for_this_file=False
    )

    assert cpu_calls == [], "Drizzle standard path must not invoke the warp backend"

    assert data is not None
    assert matrix_m is not None
    assert matrix_m.shape == (2, 3)
    assert mask is not None

    # returned data is the ORIGINAL prepared pixels (dark block still dark) —
    # NOT a warped/rolled image
    assert data[10, 10].sum() < 1e-3
    assert data[20, 20].sum() > 0

    # tf direction: (+4,+6) maps (20,20) -> (24,26)
    p = matrix_m @ np.array([20.0, 20.0, 1.0])
    assert abs(p[0] - 24.0) < 1e-4
    assert abs(p[1] - 26.0) < 1e-4


def test_drizzle_science_equivalence_with_deterministic_tf(tmp_path, monkeypatch):
    shape = (64, 64)
    qm_mod, obj = _drizzle_qm(shape)

    exact = SimilarityTransform(rotation=0.0, translation=(4.0, 6.0))
    monkeypatch.setattr(
        alignment_mod.aa,
        "find_transform",
        lambda source, target: (exact, ([], [])),
    )

    img = np.stack(
        [_gauss(shape, 400.0, 2.0, (20.0, 20.0))] * 3, axis=-1
    ).astype(np.float32)
    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = shape[1]
    hdr["NAXIS2"] = shape[0]
    hdr["NAXIS3"] = 3
    path = tmp_path / "in.fits"
    fits.PrimaryHDU(data=img, header=hdr).writeto(path, overwrite=True)
    ref_data = np.zeros((shape[0], shape[1], 3), dtype=np.float32)

    data, header, _scores, _wcs, matrix_m, mask = obj._process_file(
        str(path), ref_data, solve_astrometry_for_this_file=False
    )
    assert matrix_m is not None

    ok = obj._add_frame_to_drizzle_accumulators(data, header, matrix_m, mask)
    assert ok is True
    sci = obj.drizzle_accumulators[0].finalize()

    # the star (20,20) lands at (24,26) via the transform-only tf
    thr = float(np.max(sci)) * 0.5
    m = sci >= thr
    h, w = sci.shape
    yy, xx = np.indices((h, w))
    tot = sci[m].sum()
    cx = (sci[m] * xx[m]).sum() / tot
    cy = (sci[m] * yy[m]).sum() / tot
    assert abs(cx - 24.0) < 0.1
    assert abs(cy - 26.0) < 0.1


# ---------------------------------------------------------------------------
# stable target: structural pin (no reference reassignment to a stack)
# ---------------------------------------------------------------------------


def test_registration_target_never_reassigned_to_stack():
    import ast
    import pathlib

    qm_path = pathlib.Path(ROOT) / "seestar" / "queuep" / "queue_manager.py"
    tree = ast.parse(qm_path.read_text(encoding="utf-8"))
    VAR = "reference_image_data_for_global_alignment"
    reassign = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = []
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
            if VAR in names and not (
                isinstance(node.value, ast.Constant) and node.value.value is None
            ):
                call = node.value
                is_get_ref = (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "_get_reference_image"
                )
                is_flush = (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "_flush_current_batch"
                )
                if not is_get_ref and not is_flush:
                    reassign.append(node.lineno)
    assert reassign == [], (
        "RF2 requires an immutable registration target: no reassignment to a "
        f"stack may remain. Found reassignments at lines {reassign}"
    )
