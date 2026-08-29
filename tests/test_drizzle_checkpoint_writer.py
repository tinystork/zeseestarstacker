"""RSM2-D1: production atomic writer for native Drizzle checkpoints (write-only).

These tests exercise :class:`seestar.core.drizzle_checkpoint.DrizzleCheckpointWriter`
and the ``queue_manager`` safe-boundary hooks without implementing any production
reader.  They prove, in order:

A. writer roundtrip inspection — parse the manifest, verify all six
   descriptors/checksums/sizes/dtypes/shapes, ``np.load`` the arrays and compare
   bit-exactly to the owned source snapshots (including a *negative* Lanczos WHT);
B. generation atomicity / failure injection at each material stage — the prior
   ``checkpoint.json`` bytes and every referenced old artifact stay byte-identical
   and no manifest references a partial/mixed generation;
C. successful generation N->N+1 — the manifest switches atomically to only the
   N+1 descriptors, the current generation is never garbage-collected;
D. runtime safe-boundary ordering — no write after 0 frames / failed add, no
   checkpoint between channels, cadence at group_size, trailing force on
   Stop/success, idempotent force, ledger/counters match, failure aborts before
   the move;
E. canonical config/fingerprint mismatch, absent WCS/session binding, duplicate
   or unstattable source, invalid/non-finite buffers/counters all fail closed
   with no newly committed manifest.
"""

import hashlib
import json
import math
import os
import threading
from queue import Queue
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from seestar.core.drizzle_checkpoint import (
    CHECKPOINT_DIRNAME,
    MANIFEST_FILENAME,
    MANIFEST_TMP_PREFIX,
    MODE_TOKEN,
    RUN_CONFIG_FILENAME,
    SCHEMA_VERSION,
    STATE_CLEAN,
    DrizzleCheckpointError,
    DrizzleCheckpointWriter,
    build_drizzle_canonical_config,
)
from seestar.core.drizzle_core import (
    DrizzleAccumulator,
    build_output_grid,
)
from seestar.queuep.queue_manager import SeestarQueuedStacker
from seestar import run_contract


# ---------------------------------------------------------------------------
# deterministic synthetic frames / WCS
# ---------------------------------------------------------------------------

OUT_SHAPE = (32, 32)
IN_SHAPE = (24, 24)
_IH, _IW = IN_SHAPE
_YY, _XX = np.indices(IN_SHAPE, dtype=np.float64)

_FRAME_TRANSFORMS = [
    ("trans", 0.25, -0.5),
    ("trans", 6.5, 1.3),
    ("trans", -2.3, 6.8),
    ("rot", 5.0, 12.0, 12.0),
    ("trans", 3.7, -4.2),
    ("trans", 0.4, 0.6),
]


def _trans_tf(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def _rot_tf(deg, cx, cy):
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.array([cx, cy], dtype=np.float64)
    t = c - r @ c
    return np.array(
        [[r[0, 0], r[0, 1], t[0]], [r[1, 0], r[1, 1], t[1]]], dtype=np.float64
    )


def _frame_tf(spec):
    if spec[0] == "trans":
        return _trans_tf(spec[1], spec[2])
    return _rot_tf(spec[1], spec[2], spec[3])


def build_frames():
    frames = []
    for i, spec in enumerate(_FRAME_TRANSFORMS):
        tf = _frame_tf(spec)
        px = tf[0, 0] * _XX + tf[0, 1] * _YY + tf[0, 2]
        py = tf[1, 0] * _XX + tf[1, 1] * _YY + tf[1, 2]
        pixmap = np.dstack((px, py))
        in_grid = (
            (pixmap[..., 0] >= 0.0)
            & (pixmap[..., 0] < OUT_SHAPE[1])
            & (pixmap[..., 1] >= 0.0)
            & (pixmap[..., 1] < OUT_SHAPE[0])
        )
        data = (
            np.sin(_XX / 2.0 + i) * 5.0 + np.cos(_YY / 1.5) * 3.0 + 20.0
        ).astype(np.float32)
        weight = np.full(IN_SHAPE, 0.7, np.float32)
        frames.append((data, weight, pixmap, in_grid, 1.0 + 0.1 * i))
    return frames


def _add_frame_to_all(accs, data2d, weight, pixmap, in_grid, exptime):
    for acc in accs:
        acc.add(
            data2d,
            weight,
            pixmap,
            exptime=exptime,
            in_units="counts",
            in_grid_mask=in_grid,
        )


def make_wcs(shape_hw, crval=(10.0, 20.0), cdelt=(-0.001, 0.001)):
    h, w = shape_hw
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = np.array(list(cdelt))
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
    wcs.array_shape = (h, w)
    return wcs


def _file_identity(path):
    st = os.stat(path)
    return {
        "path": os.path.normcase(str(path)),
        "name": os.path.basename(str(path)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _reference_identity(tmp_path):
    p = tmp_path / "reference.fit"
    p.write_bytes(b"ref")
    return _file_identity(p)


def _identities(tmp_path, n):
    """Create ``n`` distinct source files and return their stat identities."""
    idents = []
    for i in range(n):
        p = tmp_path / f"src_{i}.fit"
        p.write_bytes(b"src-data-%d" % i)
        idents.append(_file_identity(p))
    return idents


def _plan(sources):
    return {"sources": sources, "decomposition": [len(sources)]}


def _session_binding(tmp_path, plan):
    return {
        "input_roots": [str(tmp_path)],
        "reference": _reference_identity(tmp_path),
        "plan": plan,
    }


def _counters(frame_count):
    return {
        "frame_count": frame_count,
        "stacked_batches_count": frame_count,
        "total_exposure_seconds": float(frame_count),
        "exposure_unknown_count": 0,
        "exposure_min": 1.0 if frame_count else None,
        "exposure_max": 1.0 if frame_count else None,
    }


def _writer(tmp_path, out_shape=OUT_SHAPE, kernel="square"):
    wcs = make_wcs(out_shape)
    out_wcs, out_shape_hw = build_output_grid(wcs, out_shape, 1.0)

    class _Qm:
        pass

    qm = _Qm()
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
    qm.drizzle_scale = 1.0
    qm.drizzle_kernel = kernel
    qm.drizzle_pixfrac = 1.0
    qm.drizzle_wht_threshold_effective = 0.0
    qm.drizzle_fillval = "0.0"

    cfg = build_drizzle_canonical_config(qm, product_version="8.2.0")
    return (
        DrizzleCheckpointWriter(
            str(tmp_path),
            "8.2.0",
            cfg,
            out_wcs,
            out_shape_hw,
        ),
        out_wcs,
        out_shape_hw,
    )


def _accs(out_shape, kernel="square"):
    return [
        DrizzleAccumulator(out_shape, kernel=kernel, pixfrac=1.0)
        for _ in range(3)
    ]


def _read_manifest(tmp_path):
    return json.loads(
        (Path(tmp_path) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )


# ---------------------------------------------------------------------------
# A. writer roundtrip inspection (bit-exact, incl. negative Lanczos WHT)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["square", "lanczos2"])
def test_writer_roundtrip_inspection_bit_exact(tmp_path, kernel):
    writer, _wcs, out_shape_hw = _writer(tmp_path, kernel=kernel)
    accs = _accs(out_shape_hw, kernel=kernel)
    frames = build_frames()
    for (data, weight, pixmap, in_grid, exptime) in frames[:4]:
        _add_frame_to_all(accs, data, weight, pixmap, in_grid, exptime)

    # Owned source snapshots taken BEFORE commit (never aliased afterwards).
    snapshots = [(a._out_img.copy(), a._out_wht.copy()) for a in accs]

    idents = _identities(tmp_path, 4)
    plan = _plan(idents)
    binding = {
        "input_roots": [str(tmp_path)],
        "reference": idents[0],
        "plan": plan,
    }
    gen = writer.commit(
        accs,
        session_binding=binding,
        counters=_counters(4),
        completed_sources=idents,
    )
    assert gen == 1

    manifest = _read_manifest(tmp_path)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["mode"] == MODE_TOKEN
    assert manifest["state"] == STATE_CLEAN
    assert manifest["generation"] == 1
    assert manifest["output_shape_hw"] == list(out_shape_hw)
    assert manifest["frame_count"] == 4
    assert manifest["stacked_batches_count"] == 4

    # All six descriptors present, exact sizes / checksums / dtype / shape.
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    assert len(manifest["channels"]) == 3
    for ch in manifest["channels"]:
        c = ch["channel"]
        assert 0 <= c <= 2
        for kind in ("out_img", "out_wht"):
            desc = ch[kind]
            path = ckpt_dir / desc["file"]
            assert path.is_file()
            raw = path.read_bytes()
            assert desc["size"] == len(raw)
            assert desc["sha256"] == hashlib.sha256(raw).hexdigest()
            assert desc["dtype"] == "float32"
            assert desc["shape"] == list(out_shape_hw)
            loaded = np.load(path)
            assert loaded.dtype == np.float32
            assert loaded.shape == tuple(out_shape_hw)
            assert np.array_equal(loaded, snapshots[c][0 if kind == "out_img" else 1])

    if kernel == "lanczos2":
        # The deterministic geometry must actually produce signed WHT, and the
        # persisted arrays must preserve every negative sample bit-exactly.
        neg = np.any(snapshots[0][1] < 0.0)
        assert neg, "lanczos2 geometry did not create a negative WHT sample"
        loaded_wht = np.load(ckpt_dir / manifest["channels"][0]["out_wht"]["file"])
        assert np.array_equal(loaded_wht, snapshots[0][1])

    # canonical config evidence is present and self-consistent.
    cfg_path = Path(tmp_path) / RUN_CONFIG_FILENAME
    assert cfg_path.is_file()
    report = run_contract.read_cfg(str(cfg_path))
    assert report.config.full_digest() == manifest["run_config_digest"]
    assert report.config.drizzle_fingerprint() == manifest["scientific_fingerprint"]
    assert report.config.scientific == manifest["scientific_config"]


def test_writer_refuses_empty_checkpoint(tmp_path):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    source_ident = _reference_identity(tmp_path)
    binding = _session_binding(tmp_path, _plan([source_ident]))
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs,
            session_binding=binding,
            counters=_counters(0),
            completed_sources=[],
        )
    # No manifest / generation published.
    assert not (Path(tmp_path) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).exists()


# ---------------------------------------------------------------------------
# B. generation atomicity / failure injection at each material stage
# ---------------------------------------------------------------------------


def _commit_gen1(tmp_path, kernel="square"):
    writer, _wcs, out_shape_hw = _writer(tmp_path, kernel=kernel)
    accs = _accs(out_shape_hw, kernel=kernel)
    frames = build_frames()
    for (data, weight, pixmap, in_grid, exptime) in frames[:2]:
        _add_frame_to_all(accs, data, weight, pixmap, in_grid, exptime)
    idents = _identities(tmp_path, 4)
    binding = _session_binding(tmp_path, _plan(idents))
    gen = writer.commit(
        accs, session_binding=binding, counters=_counters(2),
        completed_sources=idents[:2],
    )
    assert gen == 1
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    manifest_bytes = (ckpt_dir / MANIFEST_FILENAME).read_bytes()
    artifacts = {}
    for name in sorted(os.listdir(ckpt_dir)):
        if name.endswith(".npy"):
            artifacts[name] = (ckpt_dir / name).read_bytes()
    return writer, accs, binding, idents, manifest_bytes, artifacts


def _add_more(accs, n_more):
    frames = build_frames()
    for (data, weight, pixmap, in_grid, exptime) in frames[2 : 2 + n_more]:
        _add_frame_to_all(accs, data, weight, pixmap, in_grid, exptime)


def _attempt_gen2(writer, accs, binding, idents, n_total):
    return writer.commit(
        accs, session_binding=binding, counters=_counters(n_total),
        completed_sources=idents[:n_total],
    )


def _assert_gen1_intact(tmp_path, manifest_bytes, artifacts):
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    assert (ckpt_dir / MANIFEST_FILENAME).read_bytes() == manifest_bytes
    for name, data in artifacts.items():
        assert (ckpt_dir / name).exists(), f"{name} missing after failure"
        assert (ckpt_dir / name).read_bytes() == data


def test_failure_at_array_stage_preserves_prior_generation(tmp_path, monkeypatch):
    writer, accs, binding, idents, mbytes, artifacts = _commit_gen1(tmp_path)
    _add_more(accs, 2)

    calls = {"n": 0}
    orig = writer._write_array_artifact

    def failing(name, arr):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("injected array failure")
        return orig(name, arr)

    monkeypatch.setattr(writer, "_write_array_artifact", failing)
    with pytest.raises(DrizzleCheckpointError):
        _attempt_gen2(writer, accs, binding, idents, 4)

    _assert_gen1_intact(tmp_path, mbytes, artifacts)
    # No gen-2 files (attempt cleanup), no manifest referencing a mixed gen.
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    assert not any("gen-00000002" in n for n in os.listdir(ckpt_dir))


def test_failure_at_cfg_stage_preserves_prior_generation(tmp_path, monkeypatch):
    writer, accs, binding, idents, mbytes, artifacts = _commit_gen1(tmp_path)
    _add_more(accs, 2)

    def fail_cfg():
        raise RuntimeError("injected cfg failure")

    monkeypatch.setattr(run_contract, "write_cfg", fail_cfg)
    with pytest.raises(DrizzleCheckpointError):
        _attempt_gen2(writer, accs, binding, idents, 4)

    _assert_gen1_intact(tmp_path, mbytes, artifacts)
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    assert not any("gen-00000002" in n for n in os.listdir(ckpt_dir))


def test_failure_at_manifest_tmp_write_preserves_prior_generation(tmp_path, monkeypatch):
    writer, accs, binding, idents, mbytes, artifacts = _commit_gen1(tmp_path)
    _add_more(accs, 2)

    def fail_manifest(manifest):
        raise RuntimeError("injected manifest write failure")

    monkeypatch.setattr(writer, "_write_manifest", fail_manifest)
    with pytest.raises(DrizzleCheckpointError):
        _attempt_gen2(writer, accs, binding, idents, 4)

    _assert_gen1_intact(tmp_path, mbytes, artifacts)
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    assert not any("gen-00000002" in n for n in os.listdir(ckpt_dir))


def test_failure_at_manifest_replace_preserves_prior_generation(tmp_path, monkeypatch):
    writer, accs, binding, idents, mbytes, artifacts = _commit_gen1(tmp_path)
    _add_more(accs, 2)

    real_replace = os.replace

    def replace_raises_on_manifest(src_p, dst_p):
        if str(dst_p).endswith(MANIFEST_FILENAME):
            raise RuntimeError("injected replace failure")
        return real_replace(src_p, dst_p)

    monkeypatch.setattr(os, "replace", replace_raises_on_manifest)
    with pytest.raises(DrizzleCheckpointError):
        _attempt_gen2(writer, accs, binding, idents, 4)

    _assert_gen1_intact(tmp_path, mbytes, artifacts)
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    assert not any("gen-00000002" in n for n in os.listdir(ckpt_dir))


def test_failure_at_array_fsync_preserves_prior_generation(tmp_path, monkeypatch):
    writer, accs, binding, idents, mbytes, artifacts = _commit_gen1(tmp_path)
    _add_more(accs, 2)

    def fail_fsync(fd):
        raise RuntimeError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(DrizzleCheckpointError):
        _attempt_gen2(writer, accs, binding, idents, 4)

    _assert_gen1_intact(tmp_path, mbytes, artifacts)


# Restart / namespace safety: a second writer on the same output must refuse
# before any write and must never reuse/overwrite/clean/GC a prior generation.

def test_second_writer_refuses_existing_checkpoint(tmp_path):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    frames = build_frames()
    for (data, weight, pixmap, in_grid, exptime) in frames[:2]:
        _add_frame_to_all(accs, data, weight, pixmap, in_grid, exptime)
    idents = _identities(tmp_path, 2)
    binding = _session_binding(tmp_path, _plan(idents))
    assert writer.commit(
        accs, session_binding=binding, counters=_counters(2),
        completed_sources=idents,
    ) == 1

    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    manifest_bytes = (ckpt_dir / MANIFEST_FILENAME).read_bytes()
    artifacts = {
        n: (ckpt_dir / n).read_bytes()
        for n in sorted(os.listdir(ckpt_dir))
        if n.endswith(".npy")
    }
    assert len(artifacts) == 6

    # A second writer on the same output must refuse at construction, before
    # any array write, leaving every prior byte identical.
    with pytest.raises(DrizzleCheckpointError):
        _writer(tmp_path)

    assert (ckpt_dir / MANIFEST_FILENAME).read_bytes() == manifest_bytes
    for n, data in artifacts.items():
        assert (ckpt_dir / n).exists(), f"{n} missing after second writer"
        assert (ckpt_dir / n).read_bytes() == data


def test_first_commit_refuses_namespace_materialized_after_construction(tmp_path):
    # Both writers construct while the namespace is still empty.
    writer_b, _wcs_b, out_shape_b = _writer(tmp_path)
    writer_a, _wcs_a, out_shape_a = _writer(tmp_path)

    accs_a = _accs(out_shape_a)
    frames = build_frames()
    for (data, weight, pixmap, in_grid, exptime) in frames[:2]:
        _add_frame_to_all(accs_a, data, weight, pixmap, in_grid, exptime)
    idents = _identities(tmp_path, 2)
    binding = _session_binding(tmp_path, _plan(idents))
    assert writer_a.commit(
        accs_a, session_binding=binding, counters=_counters(2),
        completed_sources=idents,
    ) == 1

    # writer_b's first commit must now refuse (namespace is no longer empty).
    accs_b = _accs(out_shape_b)
    with pytest.raises(DrizzleCheckpointError):
        writer_b.commit(
            accs_b, session_binding=binding, counters=_counters(2),
            completed_sources=idents,
        )


# ---------------------------------------------------------------------------
# C. successful generation N -> N+1
# ---------------------------------------------------------------------------


def test_generation_n_to_n1_switches_atomically_and_keeps_current(tmp_path):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    frames = build_frames()
    idents = _identities(tmp_path, 4)
    binding = _session_binding(tmp_path, _plan(idents))

    for (data, weight, pixmap, in_grid, exptime) in frames[:2]:
        _add_frame_to_all(accs, data, weight, pixmap, in_grid, exptime)
    g1 = writer.commit(
        accs, session_binding=binding, counters=_counters(2),
        completed_sources=idents[:2],
    )
    assert g1 == 1

    snap1 = [a._out_img.copy() for a in accs]
    for (data, weight, pixmap, in_grid, exptime) in frames[2:4]:
        _add_frame_to_all(accs, data, weight, pixmap, in_grid, exptime)
    snap2 = [a._out_img.copy() for a in accs]
    g2 = writer.commit(
        accs, session_binding=binding, counters=_counters(4),
        completed_sources=idents[:4],
    )
    assert g2 == 2

    manifest = _read_manifest(tmp_path)
    assert manifest["generation"] == 2
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME

    # Manifest references only generation-2 artifacts.
    gen2_names = {
        d["file"] for ch in manifest["channels"] for d in (ch["out_img"], ch["out_wht"])
    }
    for name in gen2_names:
        assert "-00000002-" in name
    for ch in manifest["channels"]:
        loaded_img = np.load(ckpt_dir / ch["out_img"]["file"])
        assert np.array_equal(loaded_img, snap2[ch["channel"]])

    # Current generation (2) never GC'd.
    for name in gen2_names:
        assert (ckpt_dir / name).exists()
    # Stale generation (1) best-effort GC'd.
    assert not any("gen-00000001" in n for n in os.listdir(ckpt_dir))
    # The two generations were scientifically distinct.
    assert not np.array_equal(snap1[0], snap2[0])


# ---------------------------------------------------------------------------
# D. runtime safe-boundary ordering (via queue_manager hooks)
# ---------------------------------------------------------------------------


def _build_qm(tmp_path, group_size):
    qm = object.__new__(SeestarQueuedStacker)
    qm.drizzle_active_session = True
    qm.is_mosaic_run = False
    qm.reference_wcs_object = make_wcs((32, 32))
    qm.drizzle_output_wcs, out_shape = build_output_grid(
        qm.reference_wcs_object, (32, 32), 1.0
    )
    qm.drizzle_output_shape_hw = out_shape
    qm.drizzle_accumulators = _accs(out_shape)
    qm.drizzle_group_size = group_size
    qm.drizzle_processing_policy = "standard"
    qm._drizzle_frame_count = 0
    qm._drizzle_group_index = 0
    qm.stacked_batches_count = 0
    qm.total_exposure_seconds = 0.0
    qm._exposure_unknown_count = 0
    qm._exposure_min = None
    qm._exposure_max = None
    qm.output_folder = str(tmp_path)
    qm.update_progress = lambda *a, **k: None
    qm.preview_callback = None

    qm._drizzle_checkpoint_enabled = True
    qm._drizzle_checkpoint_writer = None
    qm._drizzle_checkpoint_plan = None
    qm._drizzle_completed_sources = []
    qm._drizzle_checkpoint_last_committed_frames = 0

    qm._resume_input_roots = [str(tmp_path)]
    qm._resume_reference_identity = _reference_identity(tmp_path)

    # canonical config fields
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
    qm.drizzle_scale = 1.0
    qm.drizzle_kernel = "square"
    qm.drizzle_pixfrac = 1.0
    qm.drizzle_wht_threshold_effective = 0.0
    qm.drizzle_fillval = "0.0"
    return qm, out_shape


def _init_qm_writer(qm, source_paths=None):
    cfg = build_drizzle_canonical_config(
        qm, product_version=qm._canonical_product_version()
    )
    qm._drizzle_checkpoint_writer = DrizzleCheckpointWriter(
        qm.output_folder,
        qm._canonical_product_version(),
        cfg,
        qm.drizzle_output_wcs,
        qm.drizzle_output_shape_hw,
    )
    if source_paths is None:
        idents = [_reference_identity(Path(qm.output_folder))]
    else:
        idents = [_file_identity(Path(p)) for p in source_paths]
    qm._drizzle_checkpoint_plan = _plan(idents)


def test_init_drizzle_checkpoint_binds_plan_and_writer(tmp_path):
    qm, _ = _build_qm(tmp_path, group_size=2)
    qm.queue = Queue()
    src_paths = [_source_file(tmp_path, i) for i in range(3)]
    for p in src_paths:
        qm.queue.put(p)

    assert qm._init_drizzle_checkpoint() is True
    assert qm._drizzle_checkpoint_writer is not None
    assert len(qm._drizzle_checkpoint_plan["sources"]) == 3
    # The plan is the exact ordered source identities captured from the queue.
    assert [s["name"] for s in qm._drizzle_checkpoint_plan["sources"]] == [
        f"src_{i}.fit" for i in range(3)
    ]


def _source_file(tmp_path, i):
    p = tmp_path / f"src_{i}.fit"
    p.write_bytes(b"data-%d" % i)
    return str(p)


def test_cadence_and_trailing_flush_ordering(tmp_path):
    qm, _ = _build_qm(tmp_path, group_size=2)
    source_paths = [_source_file(tmp_path, i) for i in range(5)]
    _init_qm_writer(qm, source_paths)
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME

    # 0 frames: force flush must be a no-op (no checkpoint published).
    qm._drizzle_checkpoint_force_flush()
    assert not (ckpt_dir / MANIFEST_FILENAME).exists()

    for i in range(5):
        src = source_paths[i]
        data, weight, pixmap, in_grid, exptime = build_frames()[i]
        data_hwc = np.stack([data, data, data], axis=-1).astype(np.float32)
        hdr = fits.Header()
        hdr["EXPTIME"] = 1.0
        tf = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        ok = qm._add_frame_to_drizzle_accumulators(
            data_hwc, hdr, tf, weight, native_wcs=None
        )
        assert ok is True
        qm._drizzle_group_tick()
        qm._admit_exposure(1.0, 0, 1.0, 1.0)
        qm.stacked_batches_count += 1
        qm._drizzle_checkpoint_after_frame(src)

    # group_size=2: commits at frames 2 and 4 (not at 5).
    manifest = _read_manifest(tmp_path)
    assert manifest["frame_count"] == 4
    assert len(qm._drizzle_completed_sources) == 5
    assert qm._drizzle_checkpoint_last_committed_frames == 4

    # Trailing force flush commits the partial group (frame 5).
    qm._drizzle_checkpoint_force_flush()
    manifest = _read_manifest(tmp_path)
    assert manifest["frame_count"] == 5
    assert manifest["generation"] == 3
    assert manifest["stacked_batches_count"] == 5
    assert len(manifest["completed_sources"]) == 5

    # Idempotent: a second force flush does not publish a new generation.
    qm._drizzle_checkpoint_force_flush()
    manifest2 = _read_manifest(tmp_path)
    assert manifest2["generation"] == 3
    assert manifest2 == manifest


def test_no_checkpoint_on_failed_add(tmp_path):
    qm, _ = _build_qm(tmp_path, group_size=1)
    _init_qm_writer(qm)
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME

    # A failed add (tf=None AND no native WCS -> the guard returns False) must
    # not checkpoint: no source recorded, no manifest published.
    data, weight, pixmap, in_grid, exptime = build_frames()[0]
    data_hwc = np.stack([data, data, data], axis=-1).astype(np.float32)
    hdr = fits.Header()
    ok = qm._add_frame_to_drizzle_accumulators(
        data_hwc, hdr, None, weight, native_wcs=None
    )
    assert ok is False
    assert qm._drizzle_frame_count == 0
    assert qm._drizzle_completed_sources == []
    assert not (ckpt_dir / MANIFEST_FILENAME).exists()


def test_commit_failure_aborts_before_move(tmp_path, monkeypatch):
    qm, _ = _build_qm(tmp_path, group_size=1)
    src = _source_file(tmp_path, 0)
    _init_qm_writer(qm, [src])

    # Accept one frame and record the accepted counters/exposure.
    data, weight, pixmap, in_grid, exptime = build_frames()[0]
    data_hwc = np.stack([data, data, data], axis=-1).astype(np.float32)
    hdr = fits.Header()
    hdr["EXPTIME"] = 1.0
    tf = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    assert qm._add_frame_to_drizzle_accumulators(data_hwc, hdr, tf, weight) is True
    qm._drizzle_group_tick()
    qm._admit_exposure(1.0, 0, 1.0, 1.0)
    qm.stacked_batches_count += 1

    # Inject a persistence failure at the writer commit.
    def fail_commit(*a, **k):
        raise DrizzleCheckpointError("injected commit failure")

    monkeypatch.setattr(qm._drizzle_checkpoint_writer, "commit", fail_commit)

    with pytest.raises(DrizzleCheckpointError):
        qm._drizzle_checkpoint_after_frame(src)

    # No manifest published, and the source was NOT moved.
    assert not (Path(tmp_path) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).exists()
    assert Path(src).exists()


# ---------------------------------------------------------------------------
# E. fail-closed validation (no newly committed manifest)
# ---------------------------------------------------------------------------


def test_commit_fails_closed_without_manifest(tmp_path):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    src = _reference_identity(tmp_path)
    binding = _session_binding(tmp_path, _plan([src]))
    frames = build_frames()
    _add_frame_to_all(accs, *frames[0])

    good = dict(
        session_binding=binding,
        counters=_counters(1),
        completed_sources=[src],
    )

    # 1. missing session binding (reference is None).
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs,
            session_binding={"input_roots": [str(tmp_path)], "reference": None,
                            "plan": binding["plan"]},
            counters=_counters(1),
            completed_sources=[src],
        )

    # 2. absent WCS (constructed writer already fails; test serialize helper).
    from seestar.core.drizzle_checkpoint import serialize_wcs_header
    with pytest.raises(DrizzleCheckpointError):
        serialize_wcs_header(None)

    # 3. duplicate source in ledger.
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding, counters=_counters(1),
            completed_sources=[src, src],
        )

    # 4. unstattable source in ledger (missing size / mtime_ns identity).
    bad_src = {"path": str(tmp_path / "x.fit"), "name": "x"}
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding, counters=_counters(1),
            completed_sources=[bad_src],
        )

    # 5. non-finite counters.
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding,
            counters={**_counters(1), "total_exposure_seconds": float("nan")},
            completed_sources=[src],
        )

    # 6. invalid buffer (NaN in out_img).
    accs_bad = _accs(out_shape_hw)
    _add_frame_to_all(accs_bad, *frames[0])
    accs_bad[0]._out_img[0, 0] = np.nan
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs_bad, session_binding=binding, counters=_counters(1),
            completed_sources=[src],
        )

    # 7. inconsistent per-channel total_exptime.
    accs_mismatch = _accs(out_shape_hw)
    _add_frame_to_all(accs_mismatch, *frames[0])
    accs_mismatch[1]._total_exptime = 999.0
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs_mismatch, session_binding=binding, counters=_counters(1),
            completed_sources=[src],
        )

    # No manifest was ever published.
    assert not (Path(tmp_path) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).exists()


def test_bad_canonical_config_fails_closed(tmp_path):
    """A canonical config missing an effective Drizzle field is refused at
    writer construction (no checkpoint namespace created)."""
    from seestar.core.drizzle_checkpoint import DrizzleCheckpointWriter
    wcs = make_wcs((16, 16))
    out_wcs, out_shape_hw = build_output_grid(wcs, (16, 16), 1.0)
    cfg = run_contract.RunConfig.from_sections(
        product_version="8.2.0",
        scientific={"drizzle_kernel_effective": "square"},
    )
    with pytest.raises(DrizzleCheckpointError):
        DrizzleCheckpointWriter(
            str(tmp_path), "8.2.0", cfg, out_wcs, out_shape_hw
        )
    assert not (Path(tmp_path) / CHECKPOINT_DIRNAME).exists()


@pytest.mark.parametrize(
    "kernel,pixfrac,fillval",
    [
        ("gaussian", 1.0, "0.0"),
        ("square", 0.5, "0.0"),
        ("square", 1.0, "1.0"),
    ],
)
def test_accumulator_deposition_mismatch_canonical_rejected(
    tmp_path, kernel, pixfrac, fillval
):
    """Writer-side preflight: D1 must never publish accumulator runtime
    deposition parameters (kernel / pixfrac / fillval) that disagree with the
    canonical ``run_config.cfg`` scientific fields.  Rejected before any
    checkpoint dir / run_config creation."""
    writer, _wcs, out_shape_hw = _writer(tmp_path)  # canonical square/1.0/"0.0"
    accs = [
        DrizzleAccumulator(
            out_shape_hw, kernel=kernel, pixfrac=pixfrac, fillval=fillval
        )
        for _ in range(3)
    ]
    _add_frame_to_all(accs, *build_frames()[0])
    src = _reference_identity(tmp_path)
    binding = _session_binding(tmp_path, _plan([src]))
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding, counters=_counters(1),
            completed_sources=[src],
        )
    _assert_no_checkpoint_published(tmp_path)


# ---------------------------------------------------------------------------
# F. manifest self-consistency (ledger == plan prefix, truthful counters)
# ---------------------------------------------------------------------------


def test_manifest_consistency_invariants(tmp_path):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    frames = build_frames()
    _add_frame_to_all(accs, *frames[0])
    idents = _identities(tmp_path, 4)

    # (a) completed ledger length != frame_count.
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=_session_binding(tmp_path, _plan(idents)),
            counters=_counters(2), completed_sources=idents[:1],
        )

    # (b) ledger is not the exact ordered plan prefix (wrong order).
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=_session_binding(tmp_path, _plan(idents)),
            counters=_counters(2), completed_sources=[idents[1], idents[0]],
        )

    # (c) stacked_batches_count != frame_count.
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=_session_binding(tmp_path, _plan(idents)),
            counters={**_counters(2), "stacked_batches_count": 1},
            completed_sources=idents[:2],
        )

    # (d) frame_count exceeds the plan length.
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=_session_binding(tmp_path, _plan(idents[:3])),
            counters=_counters(4), completed_sources=idents,
        )

    # (e) duplicate source identity in the plan.
    dup_plan = _plan([idents[0], idents[0]])
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=_session_binding(tmp_path, dup_plan),
            counters=_counters(1), completed_sources=idents[:1],
        )

    # (f) exposure_min > exposure_max / unknown_count > frame_count.
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=_session_binding(tmp_path, _plan(idents)),
            counters={**_counters(1), "exposure_min": 2.0, "exposure_max": 1.0},
            completed_sources=idents[:1],
        )
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=_session_binding(tmp_path, _plan(idents)),
            counters={**_counters(1), "exposure_unknown_count": 2},
            completed_sources=idents[:1],
        )

    # Nothing was ever published.
    assert not (Path(tmp_path) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).exists()


def test_preflight_refuses_non_json_and_nan_payload(tmp_path):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    frames = build_frames()
    _add_frame_to_all(accs, *frames[0])
    idents = _identities(tmp_path, 1)
    binding = _session_binding(tmp_path, _plan(idents))

    # NaN smuggled into the persisted scientific payload (direct-caller bypass).
    writer.scientific_config = dict(writer.scientific_config)
    writer.scientific_config["weighting_method"] = float("nan")
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding, counters=_counters(1),
            completed_sources=idents,
        )

    # Non-JSON nested value.
    writer.scientific_config["weighting_method"] = {1, 2}
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding, counters=_counters(1),
            completed_sources=idents,
        )

    # Nothing was written.
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    assert not ckpt_dir.exists() or not any(ckpt_dir.iterdir())


def test_collision_cleanup_ownership_preserves_preexisting(tmp_path):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    frames = build_frames()
    for (data, weight, pixmap, in_grid, exptime) in frames[:2]:
        _add_frame_to_all(accs, data, weight, pixmap, in_grid, exptime)
    idents = _identities(tmp_path, 4)
    binding = _session_binding(tmp_path, _plan(idents))
    assert writer.commit(
        accs, session_binding=binding, counters=_counters(2),
        completed_sources=idents[:2],
    ) == 1

    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    gen1 = {
        n: (ckpt_dir / n).read_bytes()
        for n in sorted(os.listdir(ckpt_dir))
        if "gen-00000001" in n
    }

    # Plant a pre-existing gen-2 artifact that the next attempt would collide
    # with (the 4th array written: ch1 out_wht).
    fake = ckpt_dir / "gen-00000002-ch1-out_wht.npy"
    fake.write_bytes(b"PRESERVED")
    fake_bytes = fake.read_bytes()

    _add_more(accs, 2)
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding, counters=_counters(4),
            completed_sources=idents[:4],
        )

    # The pre-existing gen-2 artifact is untouched; no other gen-2 names remain
    # (only this attempt's own partial files were cleaned).
    assert fake.read_bytes() == fake_bytes
    assert not any(
        "gen-00000002" in n and n != fake.name for n in os.listdir(ckpt_dir)
    )
    # gen1 artifacts + manifest intact.
    for n, data in gen1.items():
        assert (ckpt_dir / n).read_bytes() == data


# ---------------------------------------------------------------------------
# G. manifest temp ownership (explicit per-attempt ownership)
# ---------------------------------------------------------------------------


def test_cleanup_never_removes_foreign_manifest_temp(tmp_path, monkeypatch):
    """An attempt's cleanup removes only its own manifest temp, never a foreign
    one (explicit temp ownership; invariant #8)."""
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    idents = _identities(tmp_path, 4)
    binding = _session_binding(tmp_path, _plan(idents))

    frames = build_frames()
    for (data, weight, pixmap, in_grid, exptime) in frames[:2]:
        _add_frame_to_all(accs, data, weight, pixmap, in_grid, exptime)
    assert writer.commit(
        accs, session_binding=binding, counters=_counters(2),
        completed_sources=idents[:2],
    ) == 1

    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    manifest_bytes = (ckpt_dir / MANIFEST_FILENAME).read_bytes()

    # Plant a foreign manifest temp (as if another writer/attempt owned it).
    foreign = ckpt_dir / f"{MANIFEST_TMP_PREFIX}1111.0.deadbeef"
    foreign.write_bytes(b"FOREIGN-MANIFEST-TEMP")
    foreign_bytes = foreign.read_bytes()

    # Add more frames, then fail at the manifest *replace* stage so this
    # attempt creates (and owns) its own unique temp before failing/cleaning up.
    _add_more(accs, 2)
    real_replace = os.replace

    def replace_fail(src, dst):
        if str(dst).endswith(MANIFEST_FILENAME):
            raise RuntimeError("injected replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_fail)
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding, counters=_counters(4),
            completed_sources=idents[:4],
        )

    # The foreign temp is byte-identical; this attempt's own temp is gone.
    assert foreign.exists()
    assert foreign.read_bytes() == foreign_bytes
    own_temps = [
        n for n in os.listdir(ckpt_dir)
        if n.startswith(MANIFEST_TMP_PREFIX) and n != foreign.name
    ]
    assert own_temps == []

    # Prior manifest still committed and byte-identical.
    assert (ckpt_dir / MANIFEST_FILENAME).read_bytes() == manifest_bytes


def test_concurrent_writers_winner_publishes_despite_loser_cleanup(tmp_path):
    """Deterministic two-writer race: both pass the empty-namespace check, the
    winner owns the manifest temp, the loser fails on artifact collision and
    cleans up, yet the winner still publishes (invariant #7)."""
    from seestar.core.drizzle_checkpoint import _fsync_dir

    writer_winner, _wcs_w, shape = _writer(tmp_path)
    writer_loser, _wcs_l, _ = _writer(tmp_path)

    accs_winner = _accs(shape)
    accs_loser = _accs(shape)
    idents = _identities(tmp_path, 2)
    binding = _session_binding(tmp_path, _plan(idents))
    counters = _counters(2)

    frames = build_frames()
    for (data, weight, pixmap, in_grid, exptime) in frames[:2]:
        _add_frame_to_all(accs_winner, data, weight, pixmap, in_grid, exptime)
        _add_frame_to_all(accs_loser, data, weight, pixmap, in_grid, exptime)

    # Coordination primitives.
    check_barrier = threading.Barrier(2)
    winner_temp_ready = threading.Event()
    loser_failed = threading.Event()

    # 1. Both writers must pass the first-commit namespace check while the
    #    namespace is still empty (barrier synchronizes the two checks).
    orig_refuse_w = writer_winner._refuse_existing_checkpoint
    orig_refuse_l = writer_loser._refuse_existing_checkpoint

    def refuse_w():
        orig_refuse_w()
        check_barrier.wait(timeout=30)

    def refuse_l():
        orig_refuse_l()
        check_barrier.wait(timeout=30)

    writer_winner._refuse_existing_checkpoint = refuse_w
    writer_loser._refuse_existing_checkpoint = refuse_l

    # 2. Winner: write its owned manifest temp, then pause before the replace
    #    until the loser has failed and cleaned up.
    def winner_write_manifest(manifest):
        os.makedirs(writer_winner._dir, exist_ok=True)
        payload = json.dumps(
            manifest, sort_keys=True, indent=2, ensure_ascii=False,
            allow_nan=False,
        )
        tmp = writer_winner._claim_manifest_temp(payload)
        winner_temp_ready.set()
        assert loser_failed.wait(timeout=30), "loser did not fail/clean up in time"
        os.replace(tmp, os.path.join(writer_winner._dir, MANIFEST_FILENAME))
        writer_winner._manifest_tmp_path = None
        _fsync_dir(writer_winner._dir)
        return True

    writer_winner._write_manifest = winner_write_manifest

    # 3. Loser: block before its first artifact write until the winner has
    #    written its manifest temp, then collide on the O_EXCL artifact claim.
    orig_loser_write = writer_loser._write_array_artifact

    def loser_write_artifact(arr, final_name):
        assert winner_temp_ready.wait(timeout=30), "winner temp not ready in time"
        return orig_loser_write(arr, final_name)

    writer_loser._write_array_artifact = loser_write_artifact

    results = {}

    def loser_run():
        try:
            writer_loser.commit(
                accs_loser, session_binding=binding, counters=counters,
                completed_sources=idents,
            )
            results["loser"] = "committed"
        except DrizzleCheckpointError as exc:
            results["loser"] = f"refused: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface any surprise
            results["loser"] = f"unexpected: {exc!r}"
        finally:
            loser_failed.set()

    def winner_run():
        try:
            results["winner_gen"] = writer_winner.commit(
                accs_winner, session_binding=binding, counters=counters,
                completed_sources=idents,
            )
            results["winner"] = "committed"
        except Exception as exc:  # noqa: BLE001 - capture any failure
            results["winner"] = f"failed: {exc!r}"

    loser_thread = threading.Thread(target=loser_run)
    winner_thread = threading.Thread(target=winner_run)
    loser_thread.start()
    winner_thread.start()
    loser_thread.join(timeout=60)
    winner_thread.join(timeout=60)

    assert not loser_thread.is_alive(), "loser thread hung"
    assert not winner_thread.is_alive(), "winner thread hung"

    # The loser must have failed (artifact collision) and cleaned up; the
    # winner must have committed generation 1.
    assert results["loser"].startswith("refused:"), results.get("loser")
    # The loser failed specifically on the O_EXCL artifact collision, not on a
    # namespace refusal (both had already passed the empty-namespace check).
    assert "already exists" in results["loser"], results.get("loser")
    assert results["winner"] == "committed", results.get("winner")
    assert results["winner_gen"] == 1

    # Winner's manifest + six artifacts are valid and byte-consistent.
    manifest = _read_manifest(tmp_path)
    assert manifest["generation"] == 1
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    for ch in manifest["channels"]:
        c = ch["channel"]
        for kind in ("out_img", "out_wht"):
            desc = ch[kind]
            p = ckpt_dir / desc["file"]
            assert p.is_file()
            raw = p.read_bytes()
            assert desc["size"] == len(raw)
            assert desc["sha256"] == hashlib.sha256(raw).hexdigest()
            expected = accs_winner[c]._out_img if kind == "out_img" else accs_winner[c]._out_wht
            assert np.array_equal(np.load(p), expected)

    # No stray owned manifest temp remains after the winner committed.
    assert not any(
        n.startswith(MANIFEST_TMP_PREFIX) for n in os.listdir(ckpt_dir)
    )


# ---------------------------------------------------------------------------
# H. adversarial fail-closed validation (strict integer/float boundary)
# ---------------------------------------------------------------------------


def _assert_no_checkpoint_published(tmp_path):
    """No checkpoint artifact, manifest, or run_config was ever written."""
    ckpt_dir = Path(tmp_path) / CHECKPOINT_DIRNAME
    assert not (ckpt_dir / MANIFEST_FILENAME).exists()
    assert not ckpt_dir.exists() or not any(ckpt_dir.iterdir())
    assert not (Path(tmp_path) / RUN_CONFIG_FILENAME).exists()


@pytest.mark.parametrize("bad", [42, "abc", [1, 2], object()])
def test_malformed_session_binding_type_fails_closed(tmp_path, bad):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    _add_frame_to_all(accs, *build_frames()[0])
    src = _reference_identity(tmp_path)
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=bad, counters=_counters(1),
            completed_sources=[src],
        )
    _assert_no_checkpoint_published(tmp_path)


@pytest.mark.parametrize("bad", [42, "abc", {1, 2}, object()])
def test_malformed_completed_sources_type_fails_closed(tmp_path, bad):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    _add_frame_to_all(accs, *build_frames()[0])
    src = _reference_identity(tmp_path)
    binding = _session_binding(tmp_path, _plan([src]))
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding, counters=_counters(1),
            completed_sources=bad,
        )
    _assert_no_checkpoint_published(tmp_path)


@pytest.mark.parametrize("bad_counters", [None, [], "abc", 42])
def test_non_mapping_counters_fails_closed(tmp_path, bad_counters):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    _add_frame_to_all(accs, *build_frames()[0])
    src = _reference_identity(tmp_path)
    binding = _session_binding(tmp_path, _plan([src]))
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding, counters=bad_counters,
            completed_sources=[src],
        )
    _assert_no_checkpoint_published(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_count", 1.5),
        ("frame_count", "1"),
        ("frame_count", True),
        ("frame_count", np.float64(1.0)),
        ("frame_count", None),
        ("stacked_batches_count", 1.5),
        ("stacked_batches_count", "1"),
        ("stacked_batches_count", True),
        ("exposure_unknown_count", 0.5),
        ("exposure_unknown_count", "0"),
        ("exposure_unknown_count", True),
        ("exposure_unknown_count", []),
    ],
)
def test_non_integral_or_bool_counters_fail_closed(tmp_path, field, value):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    _add_frame_to_all(accs, *build_frames()[0])
    src = _reference_identity(tmp_path)
    binding = _session_binding(tmp_path, _plan([src]))
    counters = _counters(1)
    counters[field] = value
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding, counters=counters,
            completed_sources=[src],
        )
    _assert_no_checkpoint_published(tmp_path)


@pytest.mark.parametrize("bad", [1.5, "2", True, 0, -1])
def test_malformed_decomposition_fails_closed(tmp_path, bad):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    _add_frame_to_all(accs, *build_frames()[0])
    src = _reference_identity(tmp_path)
    binding = _session_binding(tmp_path, {"sources": [src], "decomposition": [bad]})
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding, counters=_counters(1),
            completed_sources=[src],
        )
    _assert_no_checkpoint_published(tmp_path)


@pytest.mark.parametrize(
    "attr,value",
    [
        ("pixfrac", float("nan")),
        ("pixfrac", float("inf")),
        ("pixfrac", float("-inf")),
        ("pixfrac", "not-a-number"),
        ("pixfrac", None),
        ("pixfrac", True),
        ("_total_exptime", float("nan")),
        ("_total_exptime", float("inf")),
        ("_total_exptime", "abc"),
        ("_total_exptime", None),
        ("_total_exptime", -1.0),
    ],
)
def test_malformed_accumulator_float_fails_closed(tmp_path, attr, value):
    writer, _wcs, out_shape_hw = _writer(tmp_path)
    accs = _accs(out_shape_hw)
    _add_frame_to_all(accs, *build_frames()[0])
    setattr(accs[0], attr, value)
    src = _reference_identity(tmp_path)
    binding = _session_binding(tmp_path, _plan([src]))
    with pytest.raises(DrizzleCheckpointError):
        writer.commit(
            accs, session_binding=binding, counters=_counters(1),
            completed_sources=[src],
        )
    _assert_no_checkpoint_published(tmp_path)
