"""RSM2-D2A: read-only loader/validator for the native Drizzle checkpoint.

These tests exercise :func:`seestar.core.drizzle_checkpoint.read_drizzle_checkpoint`
(and the :class:`DrizzleCheckpointResult` it returns) against checkpoints
produced by the D1 :class:`DrizzleCheckpointWriter`.  They prove, in order:

A. writer -> reader inspection roundtrip (square and *signed-WHT* lanczos2):
   the three accumulators are reconstructed bit-exactly, the canonical config /
   WCS / counters / session / ledger are validated, ``next_source_index`` is
   exact, and a successful read never mutates the checkpoint tree.
B. continuous == write / read / reconstruct / continue, bit-identical on the
   native SCI and WHT, for ``square`` / ``gaussian`` / ``lanczos2`` at
   representative splits (group boundary 2+4, partial group 3+3, and 5+1).
C. source order / next-index exactness.
D. corruption matrix: every documented corruption class fails closed with
   :class:`DrizzleCheckpointError` and leaves the checkpoint tree byte-identical
   (snapshot before/after).
"""

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

from seestar.core.drizzle_checkpoint import (
    CHECKPOINT_DIRNAME,
    MANIFEST_FILENAME,
    RUN_CONFIG_FILENAME,
    DrizzleCheckpointError,
    DrizzleCheckpointWriter,
    build_drizzle_canonical_config,
    read_drizzle_checkpoint,
)
from seestar.core.drizzle_core import (
    DrizzleAccumulator,
    build_output_grid,
)


# ---------------------------------------------------------------------------
# deterministic synthetic frames / WCS / identity helpers
# ---------------------------------------------------------------------------

OUT_SHAPE = (32, 32)
NON_SQUARE_SHAPE = (40, 24)  # (H, W) with H != W
IN_SHAPE = (24, 24)
_IH, _IW = IN_SHAPE
_YY, _XX = np.indices(IN_SHAPE, dtype=np.float64)

FRAME_TRANSFORMS = [
    ("trans", 0.25, -0.5),
    ("trans", 6.5, 1.3),
    ("trans", -2.3, 6.8),
    ("rot", 5.0, 12.0, 12.0),
    ("trans", 3.7, -4.2),
    ("trans", 0.4, 0.6),
]
FRAME_WEIGHTS = [0.7, 0.9, 0.5, 0.8, 1.0, 0.6]
FRAME_EXPTIMES = [1.0, 2.0, 1.5, 2.5, 3.0, 1.2]

KERNELS = ["square", "gaussian", "lanczos2"]
SPLITS = [2, 3, 5]


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


def _build_frames(out_shape=OUT_SHAPE):
    frames = []
    for i, spec in enumerate(FRAME_TRANSFORMS):
        tf = _frame_tf(spec)
        px = tf[0, 0] * _XX + tf[0, 1] * _YY + tf[0, 2]
        py = tf[1, 0] * _XX + tf[1, 1] * _YY + tf[1, 2]
        pixmap = np.dstack((px, py))
        in_grid = (
            (pixmap[..., 0] >= 0.0)
            & (pixmap[..., 0] < out_shape[1])
            & (pixmap[..., 1] >= 0.0)
            & (pixmap[..., 1] < out_shape[0])
        )
        data = (
            np.sin(_XX / 2.0 + i) * 5.0 + np.cos(_YY / 1.5) * 3.0 + 20.0
        ).astype(np.float32)
        weight = np.full(IN_SHAPE, FRAME_WEIGHTS[i], np.float32)
        frames.append((data, weight, pixmap, in_grid, FRAME_EXPTIMES[i]))
    return frames


def _add_to_all(accs, frame):
    data, weight, pixmap, in_grid, exptime = frame
    for acc in accs:
        acc.add(
            data, weight, pixmap, exptime=exptime, in_units="counts",
            in_grid_mask=in_grid,
        )


def _make_wcs(shape_hw):
    h, w = shape_hw
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = np.array([-0.001, 0.001])
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
    wcs.array_shape = (h, w)
    return wcs


def _out_grid():
    return build_output_grid(_make_wcs(OUT_SHAPE), OUT_SHAPE, 1.0)


def _fake_qm(kernel):
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
    return qm


def _identity(path):
    st = os.stat(path)
    return {
        "path": os.path.normcase(str(path)),
        "name": os.path.basename(str(path)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _counters(frame_count, exptimes):
    return {
        "frame_count": frame_count,
        "stacked_batches_count": frame_count,
        "total_exposure_seconds": float(sum(exptimes)),
        "exposure_unknown_count": 0,
        "exposure_min": float(min(exptimes)),
        "exposure_max": float(max(exptimes)),
    }


def _write_checkpoint(tmp_path, kernel="square", n_sources=4, frame_count=2,
                      out_shape=OUT_SHAPE):
    """Write a valid native Drizzle checkpoint and return its context."""
    out_wcs, out_shape_hw = build_output_grid(_make_wcs(out_shape), out_shape, 1.0)
    cfg = build_drizzle_canonical_config(_fake_qm(kernel), product_version="8.2.0")
    writer = DrizzleCheckpointWriter(
        str(tmp_path), "8.2.0", cfg, out_wcs, out_shape_hw
    )
    accs = [
        DrizzleAccumulator(out_shape_hw, kernel=kernel, pixfrac=1.0)
        for _ in range(3)
    ]
    frames = _build_frames(out_shape)
    for f in frames[:frame_count]:
        _add_to_all(accs, f)

    ref_path = Path(tmp_path) / "reference.fit"
    ref_path.write_bytes(b"reference-bytes")
    ref_ident = _identity(ref_path)

    src_paths = []
    for i in range(n_sources):
        p = Path(tmp_path) / f"src_{i}.fit"
        p.write_bytes(b"source-data-%d" % i)
        src_paths.append(p)
    src_idents = [_identity(p) for p in src_paths]

    binding = {
        "input_roots": [str(tmp_path)],
        "reference": ref_ident,
        "plan": {"sources": src_idents, "decomposition": [n_sources]},
    }
    gen = writer.commit(
        accs,
        session_binding=binding,
        counters=_counters(frame_count, FRAME_EXPTIMES[:frame_count]),
        completed_sources=src_idents[:frame_count],
    )
    assert gen == 1
    return {
        "writer": writer,
        "accs": accs,
        "frames": frames,
        "out_shape_hw": out_shape_hw,
        "out_wcs": out_wcs,
        "ref_path": ref_path,
        "src_paths": src_paths,
        "src_idents": src_idents,
        "frame_count": frame_count,
    }


def _tree_snapshot(root):
    """Snapshot of every regular file under ``root`` (checkpoint tree +
    run_config.cfg + source files): ``(size, mtime_ns, bytes)`` per path, so
    both byte and metadata (mtime) mutations are detected."""
    root = Path(root)
    snap = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            st = os.stat(p)
            snap[str(p.relative_to(root))] = (
                st.st_size, st.st_mtime_ns, p.read_bytes(),
            )
    return snap


def _load_manifest(tmp_path):
    p = Path(tmp_path) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME
    return json.loads(p.read_text(encoding="utf-8"))


def _save_manifest(tmp_path, m):
    p = Path(tmp_path) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME
    p.write_text(
        json.dumps(m, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _ckpt(tmp_path):
    return Path(tmp_path) / CHECKPOINT_DIRNAME


# ---------------------------------------------------------------------------
# A. writer -> reader inspection roundtrip (bit-exact, incl. signed Lanczos)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["square", "lanczos2"])
def test_writer_reader_inspection_roundtrip(kernel, tmp_path):
    ctx = _write_checkpoint(tmp_path, kernel=kernel, n_sources=4, frame_count=4)
    accs = ctx["accs"]
    before = _tree_snapshot(tmp_path)

    res = read_drizzle_checkpoint(str(tmp_path))

    # Read-only: a successful read never mutates the checkpoint tree.
    assert _tree_snapshot(tmp_path) == before

    # Result object contract.
    assert res.generation == 1
    assert res.output_shape_hw == ctx["out_shape_hw"]
    assert res.next_source_index == 4
    assert res.completed_sources == ctx["src_idents"][:4]
    assert res.session["plan"]["sources"] == ctx["src_idents"]
    assert res.counters["frame_count"] == 4
    assert res.counters["stacked_batches_count"] == 4

    # Config / digest / fingerprint self-consistency.
    assert res.config.product_version == "8.2.0"
    assert res.config.full_digest() == res.manifest["run_config_digest"]
    assert res.config.drizzle_fingerprint() == res.manifest["scientific_fingerprint"]
    assert res.config.scientific == res.manifest["scientific_config"]

    # WCS reconstructed with array_shape attached and pixel grid consistent.
    assert res.wcs.array_shape == ctx["out_shape_hw"]
    assert res.wcs.pixel_shape == (
        ctx["out_shape_hw"][1], ctx["out_shape_hw"][0]
    )

    # Three accumulators, bit-identical to the source native buffers, and the
    # per-channel runtime-effective parameters are preserved.
    assert len(res.accumulators) == 3
    for c in range(3):
        acc = res.accumulators[c]
        assert acc.kernel == kernel
        assert acc.pixfrac == 1.0
        assert acc.fillval == "0.0"
        assert acc._total_exptime == sum(FRAME_EXPTIMES[:4])
        assert np.array_equal(acc._out_img, accs[c]._out_img)
        assert np.array_equal(acc._out_wht, accs[c]._out_wht)
        assert acc._out_img.flags.owndata
        assert acc._out_wht.flags.owndata

    if kernel == "lanczos2":
        # The signed native WHT survives the writer -> reader roundtrip
        # bit-exactly, including negative samples.
        assert np.any(accs[0]._out_wht < 0.0)
        assert np.array_equal(res.accumulators[0]._out_wht, accs[0]._out_wht)


def test_non_square_wcs_axis_convention_and_continuation(tmp_path):
    """Genuinely non-square grid: ``array_shape == (H, W)``, ``pixel_shape ==
    (W, H)``, artifacts stay ``(H, W)``, and continuation is bit-identical.

    Regression for the Astropy axis-order bug: ``array_shape`` is numpy order
    ``(H, W)`` while ``pixel_shape`` is FITS order ``(W, H)``.  Square-only
    tests previously masked a reader that compared ``pixel_shape`` to
    ``output_shape_hw``.
    """
    h, w = NON_SQUARE_SHAPE
    frames = _build_frames(NON_SQUARE_SHAPE)

    # Continuous reference (all six frames, one order).
    cont = [
        DrizzleAccumulator(NON_SQUARE_SHAPE, kernel="square", pixfrac=1.0)
        for _ in range(3)
    ]
    for f in frames:
        _add_to_all(cont, f)
    cont_img = [a._out_img.copy() for a in cont]
    cont_wht = [a._out_wht.copy() for a in cont]

    ctx = _write_checkpoint(
        tmp_path, kernel="square", n_sources=6, frame_count=2,
        out_shape=NON_SQUARE_SHAPE,
    )
    res = read_drizzle_checkpoint(str(tmp_path))

    assert res.output_shape_hw == (h, w)
    assert res.wcs.array_shape == (h, w)
    assert res.wcs.pixel_shape == (w, h)
    for c in range(3):
        acc = res.accumulators[c]
        assert acc.out_shape_hw == (h, w)
        assert acc._out_img.shape == (h, w)
        assert acc._out_wht.shape == (h, w)
        assert np.array_equal(acc._out_img, ctx["accs"][c]._out_img)
        assert np.array_equal(acc._out_wht, ctx["accs"][c]._out_wht)

    # Continuation after the split must equal the uninterrupted reference.
    restored = res.accumulators
    for f in frames[2:]:
        _add_to_all(restored, f)
    for c in range(3):
        assert np.array_equal(restored[c]._out_img, cont_img[c])
        assert np.array_equal(restored[c]._out_wht, cont_wht[c])


def test_numeric_fillval_equivalent_to_canonical_string(tmp_path):
    """A numeric ``fillval=0.0`` on the accumulator is scientifically /
    serialization-equivalent to the canonical string ``"0.0"``: the writer
    preflight accepts it and the reader cross-check does not reject it."""
    out_wcs, out_shape_hw = build_output_grid(_make_wcs(OUT_SHAPE), OUT_SHAPE, 1.0)
    cfg = build_drizzle_canonical_config(_fake_qm("square"), product_version="8.2.0")
    writer = DrizzleCheckpointWriter(
        str(tmp_path), "8.2.0", cfg, out_wcs, out_shape_hw
    )
    accs = [
        DrizzleAccumulator(out_shape_hw, kernel="square", pixfrac=1.0, fillval=0.0)
        for _ in range(3)
    ]
    frames = _build_frames()
    for f in frames[:2]:
        _add_to_all(accs, f)

    ref_path = Path(tmp_path) / "reference.fit"
    ref_path.write_bytes(b"reference-bytes")
    src_paths = [Path(tmp_path) / f"src_{i}.fit" for i in range(4)]
    for i, p in enumerate(src_paths):
        p.write_bytes(b"source-data-%d" % i)
    src_idents = [_identity(p) for p in src_paths]

    binding = {
        "input_roots": [str(tmp_path)],
        "reference": _identity(ref_path),
        "plan": {"sources": src_idents, "decomposition": [4]},
    }
    assert writer.commit(
        accs,
        session_binding=binding,
        counters=_counters(2, FRAME_EXPTIMES[:2]),
        completed_sources=src_idents[:2],
    ) == 1

    res = read_drizzle_checkpoint(str(tmp_path))
    for acc in res.accumulators:
        assert float(acc.fillval) == 0.0
    # The canonical config is unchanged; only the channel fillval was numeric.
    assert res.config.scientific["drizzle_fillval"] == "0.0"


# ---------------------------------------------------------------------------
# B. continuous == write / read / reconstruct / continue (bit-identical)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("split", SPLITS)
def test_continuous_equals_write_read_reconstruct_continue(kernel, split, tmp_path):
    frames = _build_frames()

    # Continuous reference (all six frames, one order).
    cont = [DrizzleAccumulator(OUT_SHAPE, kernel=kernel, pixfrac=1.0) for _ in range(3)]
    for f in frames:
        _add_to_all(cont, f)
    cont_img = [a._out_img.copy() for a in cont]
    cont_wht = [a._out_wht.copy() for a in cont]
    cont_total = cont[0]._total_exptime

    # Prefix + checkpoint at the split boundary.
    ctx = _write_checkpoint(tmp_path, kernel=kernel, n_sources=6, frame_count=split)
    res = read_drizzle_checkpoint(str(tmp_path))
    assert res.next_source_index == split

    restored = res.accumulators
    for f in frames[split:]:
        _add_to_all(restored, f)

    for c in range(3):
        assert np.array_equal(restored[c]._out_img, cont_img[c]), (
            f"native out_img differs (kernel={kernel}, split={split}, ch={c})"
        )
        assert np.array_equal(restored[c]._out_wht, cont_wht[c]), (
            f"native out_wht differs (kernel={kernel}, split={split}, ch={c})"
        )
        assert np.array_equal(
            restored[c].finalize("divide"), cont[c].finalize("divide")
        )
        assert restored[c]._total_exptime == cont_total


# ---------------------------------------------------------------------------
# C. source order / next index exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame_count", [1, 2, 3, 4])
def test_next_source_index_exact(frame_count, tmp_path):
    ctx = _write_checkpoint(
        tmp_path, kernel="square", n_sources=4, frame_count=frame_count
    )
    res = read_drizzle_checkpoint(str(tmp_path))
    assert res.next_source_index == frame_count
    # The completed ledger is the exact ordered plan prefix.
    assert res.completed_sources == ctx["src_idents"][:frame_count]
    # The next source to resume is the first un-accumulated plan source.
    if frame_count < len(ctx["src_idents"]):
        assert res.session["plan"]["sources"][frame_count] == ctx["src_idents"][frame_count]


# ---------------------------------------------------------------------------
# D. corruption matrix (fail closed, byte-identical rejection)
# ---------------------------------------------------------------------------


def _corrupt_truncate_manifest(tmp_path):
    p = _ckpt(tmp_path) / MANIFEST_FILENAME
    p.write_bytes(p.read_bytes()[:200])


def _corrupt_manifest_nan(tmp_path):
    m = _load_manifest(tmp_path)
    m["total_exposure_seconds"] = float("nan")
    _save_manifest(tmp_path, m)


def _corrupt_unknown_schema(tmp_path):
    m = _load_manifest(tmp_path)
    m["schema_version"] = 999
    _save_manifest(tmp_path, m)


def _corrupt_wrong_mode(tmp_path):
    m = _load_manifest(tmp_path)
    m["mode"] = "classic_sumw_v1"
    _save_manifest(tmp_path, m)


def _corrupt_wrong_state(tmp_path):
    m = _load_manifest(tmp_path)
    m["state"] = "dirty"
    _save_manifest(tmp_path, m)


def _corrupt_invalid_generation(tmp_path):
    m = _load_manifest(tmp_path)
    m["generation"] = 0
    _save_manifest(tmp_path, m)


def _corrupt_config_mismatch(tmp_path):
    cfg = Path(tmp_path) / RUN_CONFIG_FILENAME
    data = json.loads(cfg.read_text(encoding="utf-8"))
    data["product_version"] = "9.9.9"
    cfg.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")


def _corrupt_fingerprint_mismatch(tmp_path):
    m = _load_manifest(tmp_path)
    m["scientific_fingerprint"] = "0" * 64
    _save_manifest(tmp_path, m)


def _corrupt_digest_mismatch(tmp_path):
    m = _load_manifest(tmp_path)
    m["run_config_digest"] = "0" * 64
    _save_manifest(tmp_path, m)


def _corrupt_version_mismatch(tmp_path):
    m = _load_manifest(tmp_path)
    m["drizzle_lib_version"] = "0.0.0"
    _save_manifest(tmp_path, m)


def _corrupt_numpy_version_mismatch(tmp_path):
    m = _load_manifest(tmp_path)
    m["numpy_version"] = "0.0.0"
    _save_manifest(tmp_path, m)


def _corrupt_wcs_empty(tmp_path):
    m = _load_manifest(tmp_path)
    m["wcs"] = {}
    _save_manifest(tmp_path, m)


def _corrupt_wcs_remove_card(tmp_path):
    m = _load_manifest(tmp_path)
    del m["wcs"]["CTYPE1"]
    _save_manifest(tmp_path, m)


def _corrupt_output_shape(tmp_path):
    m = _load_manifest(tmp_path)
    m["output_shape_hw"] = [31, 31]
    _save_manifest(tmp_path, m)


def _corrupt_channel_kernel_divergence(tmp_path):
    m = _load_manifest(tmp_path)
    for ch in m["channels"]:
        ch["kernel"] = "gaussian"
    _save_manifest(tmp_path, m)


def _corrupt_channel_pixfrac_divergence(tmp_path):
    m = _load_manifest(tmp_path)
    for ch in m["channels"]:
        ch["pixfrac"] = 0.5
    _save_manifest(tmp_path, m)


def _corrupt_channel_fillval_divergence(tmp_path):
    m = _load_manifest(tmp_path)
    for ch in m["channels"]:
        ch["fillval"] = "1.0"
    _save_manifest(tmp_path, m)


def _corrupt_traversal(tmp_path):
    m = _load_manifest(tmp_path)
    m["channels"][0]["out_img"]["file"] = "../../etc/passwd"
    _save_manifest(tmp_path, m)


def _corrupt_symlink(tmp_path):
    m = _load_manifest(tmp_path)
    desc = m["channels"][0]["out_img"]
    target = m["channels"][1]["out_img"]["file"]
    path = _ckpt(tmp_path) / desc["file"]
    path.unlink()
    path.symlink_to(target)


def _corrupt_missing_artifact(tmp_path):
    m = _load_manifest(tmp_path)
    desc = m["channels"][2]["out_wht"]
    (_ckpt(tmp_path) / desc["file"]).unlink()


def _corrupt_extra_artifact(tmp_path):
    ckpt = _ckpt(tmp_path)
    shutil.copy(
        ckpt / "gen-00000001-ch0-out_img.npy",
        ckpt / "gen-00000002-ch0-out_img.npy",
    )


def _corrupt_mixed_generation(tmp_path):
    m = _load_manifest(tmp_path)
    m["channels"][1]["out_img"]["file"] = "gen-00000002-ch1-out_img.npy"
    _save_manifest(tmp_path, m)


def _corrupt_tampered_size(tmp_path):
    m = _load_manifest(tmp_path)
    m["channels"][0]["out_img"]["size"] += 1
    _save_manifest(tmp_path, m)


def _corrupt_tampered_hash(tmp_path):
    m = _load_manifest(tmp_path)
    m["channels"][0]["out_img"]["sha256"] = "0" * 64
    _save_manifest(tmp_path, m)


def _corrupt_tampered_dtype(tmp_path):
    m = _load_manifest(tmp_path)
    m["channels"][0]["out_img"]["dtype"] = "float64"
    _save_manifest(tmp_path, m)


def _corrupt_tampered_shape(tmp_path):
    m = _load_manifest(tmp_path)
    m["channels"][0]["out_img"]["shape"] = [16, 16]
    _save_manifest(tmp_path, m)


def _corrupt_nonfinite_array(tmp_path):
    m = _load_manifest(tmp_path)
    desc = m["channels"][0]["out_img"]
    path = _ckpt(tmp_path) / desc["file"]
    arr = np.load(path).copy()
    arr.flat[0] = np.nan
    np.save(path, arr.astype(np.float32))
    raw = path.read_bytes()
    desc["size"] = len(raw)
    desc["sha256"] = hashlib.sha256(raw).hexdigest()
    _save_manifest(tmp_path, m)


def _corrupt_source_missing(tmp_path):
    (Path(tmp_path) / "src_2.fit").unlink()


def _corrupt_source_renamed(tmp_path):
    os.rename(Path(tmp_path) / "src_2.fit", Path(tmp_path) / "src_2_renamed.fit")


def _corrupt_source_size_changed(tmp_path):
    p = Path(tmp_path) / "src_2.fit"
    p.write_bytes(p.read_bytes() + b"-tampered")


def _corrupt_source_mtime_changed(tmp_path):
    p = Path(tmp_path) / "src_2.fit"
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def _corrupt_duplicate_ledger(tmp_path):
    m = _load_manifest(tmp_path)
    srcs = m["completed_sources"]
    m["completed_sources"] = [srcs[0], srcs[0]]
    _save_manifest(tmp_path, m)


def _corrupt_misaligned_ledger(tmp_path):
    m = _load_manifest(tmp_path)
    srcs = list(m["completed_sources"])
    srcs.reverse()
    m["completed_sources"] = srcs
    _save_manifest(tmp_path, m)


def _corrupt_misaligned_counters(tmp_path):
    m = _load_manifest(tmp_path)
    m["stacked_batches_count"] = m["frame_count"] + 1
    _save_manifest(tmp_path, m)


def _corrupt_unknown_count_too_high(tmp_path):
    m = _load_manifest(tmp_path)
    m["exposure_unknown_count"] = m["frame_count"] + 1
    _save_manifest(tmp_path, m)


CORRUPTIONS = [
    ("manifest_truncation", _corrupt_truncate_manifest),
    ("manifest_nan", _corrupt_manifest_nan),
    ("unknown_schema", _corrupt_unknown_schema),
    ("wrong_mode", _corrupt_wrong_mode),
    ("wrong_state", _corrupt_wrong_state),
    ("invalid_generation", _corrupt_invalid_generation),
    ("config_mismatch", _corrupt_config_mismatch),
    ("fingerprint_mismatch", _corrupt_fingerprint_mismatch),
    ("digest_mismatch", _corrupt_digest_mismatch),
    ("version_mismatch", _corrupt_version_mismatch),
    ("numpy_version_mismatch", _corrupt_numpy_version_mismatch),
    ("wcs_empty", _corrupt_wcs_empty),
    ("wcs_remove_card", _corrupt_wcs_remove_card),
    ("output_shape_mismatch", _corrupt_output_shape),
    ("channel_kernel_divergence", _corrupt_channel_kernel_divergence),
    ("channel_pixfrac_divergence", _corrupt_channel_pixfrac_divergence),
    ("channel_fillval_divergence", _corrupt_channel_fillval_divergence),
    ("path_traversal", _corrupt_traversal),
    ("symlink_artifact", _corrupt_symlink),
    ("missing_artifact", _corrupt_missing_artifact),
    ("extra_artifact", _corrupt_extra_artifact),
    ("mixed_generation", _corrupt_mixed_generation),
    ("tampered_size", _corrupt_tampered_size),
    ("tampered_hash", _corrupt_tampered_hash),
    ("tampered_dtype", _corrupt_tampered_dtype),
    ("tampered_shape", _corrupt_tampered_shape),
    ("nonfinite_array", _corrupt_nonfinite_array),
    ("source_missing", _corrupt_source_missing),
    ("source_renamed", _corrupt_source_renamed),
    ("source_size_changed", _corrupt_source_size_changed),
    ("source_mtime_changed", _corrupt_source_mtime_changed),
    ("duplicate_ledger", _corrupt_duplicate_ledger),
    ("misaligned_ledger", _corrupt_misaligned_ledger),
    ("misaligned_counters", _corrupt_misaligned_counters),
    ("unknown_count_too_high", _corrupt_unknown_count_too_high),
]


@pytest.mark.parametrize("corruption", CORRUPTIONS, ids=[c[0] for c in CORRUPTIONS])
def test_corruption_fails_closed_and_tree_byte_identical(corruption, tmp_path):
    _name, corrupt_fn = corruption
    _write_checkpoint(tmp_path, kernel="square", n_sources=4, frame_count=2)
    original = _tree_snapshot(tmp_path)

    corrupt_fn(tmp_path)
    corrupted = _tree_snapshot(tmp_path)
    # Every corruption in this matrix must actually change the persisted state
    # (so we are really proving "rejection after a real corruption").
    assert corrupted != original, f"{_name}: corruption did not change the tree"

    with pytest.raises(DrizzleCheckpointError):
        read_drizzle_checkpoint(str(tmp_path))

    # Rejection never mutates the checkpoint tree / run_config.cfg / sources.
    assert _tree_snapshot(tmp_path) == corrupted, (
        f"{_name}: reader mutated the checkpoint tree on rejection"
    )


def test_missing_checkpoint_directory_fails_closed(tmp_path):
    with pytest.raises(DrizzleCheckpointError):
        read_drizzle_checkpoint(str(tmp_path))
