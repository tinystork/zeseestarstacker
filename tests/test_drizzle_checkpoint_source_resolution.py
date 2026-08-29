"""RSM2-D2B2A: explicit, opt-in source-resolution seam for the D2A reader.

Proves the narrow source-resolution seam added after D2A/D2B1:

* the **default** :func:`read_drizzle_checkpoint` stays strict D2A — a moved /
  missing / tampered completed source or reference fails closed;
* an **opt-in** resolver (only via an explicit ``resolver=`` argument) can
  resolve a completed source / reference that was moved to the deterministic
  ``<src_dir>/<stacked_subdir>/<basename>`` destination, but the reader itself
  re-stats every returned candidate and verifies exact size + mtime_ns +
  regular-file / no-symlink — a callback claim is never trusted;
* resolution is injective and order-preserving (ambiguous / duplicated
  destinations are refused), and persisted manifest/session identities stay
  canonical (original paths) — never rewritten stacked paths;
* the shipped :class:`SafeStackedSourceResolver` is immutable and is carried as
  ``resolution_policy`` provenance; the continuation factory re-applies it on
  its fresh re-read without trusting a mutable callback and never drops source
  validation;
* square / lanczos2 continuation stays bit-exact across a move + re-arm.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

from seestar.core.drizzle_checkpoint import (
    CHECKPOINT_DIRNAME,
    MANIFEST_FILENAME,
    DrizzleCheckpointError,
    DrizzleCheckpointResult,
    DrizzleCheckpointWriter,
    DrizzleContinuation,
    SafeStackedSourceResolver,
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
FRAME_EXPTIMES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


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
        frames.append((data, weight, pixmap, in_grid, FRAME_EXPTIMES[i]))
    return frames


def _add_frame_to_all(accs, frame):
    data, weight, pixmap, in_grid, exptime = frame
    for acc in accs:
        acc.add(
            data, weight, pixmap, exptime=exptime, in_units="counts",
            in_grid_mask=in_grid,
        )


def make_wcs(shape_hw):
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


def _counters(frame_count):
    ex = FRAME_EXPTIMES[:frame_count]
    return {
        "frame_count": frame_count,
        "stacked_batches_count": frame_count,
        "total_exposure_seconds": float(sum(ex)),
        "exposure_unknown_count": 0,
        "exposure_min": float(min(ex)) if ex else None,
        "exposure_max": float(max(ex)) if ex else None,
    }


def _ckpt(tmp_path):
    return Path(tmp_path) / CHECKPOINT_DIRNAME


def _load_manifest(tmp_path):
    return json.loads(
        (_ckpt(tmp_path) / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )


def _tree_snapshot(root):
    """Byte + mtime snapshot of every regular file under ``root``."""
    root = Path(root)
    snap = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            st = os.stat(p)
            snap[str(p.relative_to(root))] = (
                st.st_size, st.st_mtime_ns, p.read_bytes(),
            )
    return snap


def _write_checkpoint(tmp_path, kernel="square", n_sources=4, frame_count=2):
    """Write a valid native Drizzle checkpoint and return its context."""
    out_wcs, out_shape_hw = build_output_grid(
        make_wcs(OUT_SHAPE), OUT_SHAPE, 1.0
    )
    cfg = build_drizzle_canonical_config(_fake_qm(kernel), product_version="8.2.0")
    writer = DrizzleCheckpointWriter(
        str(tmp_path), "8.2.0", cfg, out_wcs, out_shape_hw
    )
    accs = [
        DrizzleAccumulator(out_shape_hw, kernel=kernel, pixfrac=1.0)
        for _ in range(3)
    ]
    frames = build_frames()
    for f in frames[:frame_count]:
        _add_frame_to_all(accs, f)

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
        counters=_counters(frame_count),
        completed_sources=src_idents[:frame_count],
    )
    assert gen == 1
    return {
        "out_shape_hw": out_shape_hw,
        "out_wcs": out_wcs,
        "ref_path": ref_path,
        "ref_ident": ref_ident,
        "src_paths": src_paths,
        "src_idents": src_idents,
        "frame_count": frame_count,
    }


def _write_checkpoint_reference_in_plan(tmp_path, kernel="square", n_sources=4,
                                        frame_count=2):
    """Write a valid checkpoint whose reference is also plan source index 0.

    This is the legitimate production shape: the alignment reference may be one
    of the plan observations, so ``reference`` and ``plan["sources"][0]`` carry
    the exact same canonical identity (path + size + mtime_ns).
    """
    out_wcs, out_shape_hw = build_output_grid(
        make_wcs(OUT_SHAPE), OUT_SHAPE, 1.0
    )
    cfg = build_drizzle_canonical_config(_fake_qm(kernel), product_version="8.2.0")
    writer = DrizzleCheckpointWriter(
        str(tmp_path), "8.2.0", cfg, out_wcs, out_shape_hw
    )
    accs = [
        DrizzleAccumulator(out_shape_hw, kernel=kernel, pixfrac=1.0)
        for _ in range(3)
    ]
    frames = build_frames()
    for f in frames[:frame_count]:
        _add_frame_to_all(accs, f)

    src_paths = []
    for i in range(n_sources):
        p = Path(tmp_path) / f"src_{i}.fit"
        p.write_bytes(b"source-data-%d" % i)
        src_paths.append(p)
    src_idents = [_identity(p) for p in src_paths]

    binding = {
        "input_roots": [str(tmp_path)],
        "reference": src_idents[0],
        "plan": {"sources": src_idents, "decomposition": [n_sources]},
    }
    gen = writer.commit(
        accs,
        session_binding=binding,
        counters=_counters(frame_count),
        completed_sources=src_idents[:frame_count],
    )
    assert gen == 1
    return {
        "out_shape_hw": out_shape_hw,
        "out_wcs": out_wcs,
        "src_paths": src_paths,
        "src_idents": src_idents,
        "frame_count": frame_count,
    }


def _accs(shape, kernel="square"):
    return [
        DrizzleAccumulator(shape, kernel=kernel, pixfrac=1.0) for _ in range(3)
    ]


def _move_to_stacked(tmp_path, src_name, subdir="stacked", dup=None):
    """Simulate ``move_to_stacked`` via ``os.rename`` (preserves size+mtime)."""
    p = Path(tmp_path) / src_name
    dst_dir = Path(tmp_path) / subdir
    dst_dir.mkdir(exist_ok=True)
    if dup is not None:
        dst = dst_dir / f"{p.stem}_dup_{dup}{p.suffix}"
    else:
        dst = dst_dir / src_name
    os.rename(p, dst)
    return dst


# ---------------------------------------------------------------------------
# 1. default reader stays strict D2A
# ---------------------------------------------------------------------------


def test_default_reader_refuses_moved_completed_source(tmp_path):
    _write_checkpoint(tmp_path, n_sources=4, frame_count=2)
    _move_to_stacked(tmp_path, "src_0.fit")
    with pytest.raises(DrizzleCheckpointError):
        read_drizzle_checkpoint(str(tmp_path))


def test_default_reader_refuses_moved_reference(tmp_path):
    _write_checkpoint(tmp_path, n_sources=4, frame_count=2)
    _move_to_stacked(tmp_path, "reference.fit")
    with pytest.raises(DrizzleCheckpointError):
        read_drizzle_checkpoint(str(tmp_path))


# ---------------------------------------------------------------------------
# 2. opt-in safe resolver accepts exact move + returns ordered paths / index
# ---------------------------------------------------------------------------


def test_safe_resolver_accepts_moved_completed_and_reference(tmp_path):
    ctx = _write_checkpoint(tmp_path, n_sources=4, frame_count=2)
    _move_to_stacked(tmp_path, "reference.fit")
    _move_to_stacked(tmp_path, "src_0.fit")
    before = _tree_snapshot(tmp_path)

    resolver = SafeStackedSourceResolver("stacked")
    res = read_drizzle_checkpoint(str(tmp_path), resolver=resolver)

    # Read is read-only: the tree (sources + checkpoint) is unchanged.
    assert _tree_snapshot(tmp_path) == before

    # Resolved reference points at the moved-to-stacked counterpart.
    assert res.resolved_reference == str(
        Path(tmp_path) / "stacked" / "reference.fit"
    )
    # Ordered resolved plan paths: completed src_0 moved, others original.
    expected_plan = (
        [str(Path(tmp_path) / "stacked" / "src_0.fit")]
        + [str(Path(tmp_path) / f"src_{i}.fit") for i in range(1, 4)]
    )
    assert list(res.resolved_plan_paths) == expected_plan
    assert list(res.resolved_completed_paths) == expected_plan[:2]
    assert list(res.resolved_remaining_paths) == expected_plan[2:]
    assert res.next_source_index == 2
    assert res.resolution_policy is resolver

    # Persisted identities remain canonical (original paths, never stacked).
    assert res.completed_sources == ctx["src_idents"][:2]
    assert res.session["plan"]["sources"] == ctx["src_idents"]
    assert res.session["reference"] == ctx["ref_ident"]
    m = _load_manifest(tmp_path)
    assert m["completed_sources"] == ctx["src_idents"][:2]
    assert m["session"]["plan"]["sources"] == ctx["src_idents"]


def test_safe_resolver_mixed_pending_original_completed_moved(tmp_path):
    ctx = _write_checkpoint(tmp_path, n_sources=4, frame_count=2)
    _move_to_stacked(tmp_path, "src_0.fit")
    _move_to_stacked(tmp_path, "src_1.fit")

    res = read_drizzle_checkpoint(
        str(tmp_path), resolver=SafeStackedSourceResolver("stacked")
    )
    assert res.next_source_index == 2
    assert list(res.resolved_completed_paths) == [
        str(Path(tmp_path) / "stacked" / "src_0.fit"),
        str(Path(tmp_path) / "stacked" / "src_1.fit"),
    ]
    # Pending sources remain at their original paths.
    assert list(res.resolved_remaining_paths) == [
        str(Path(tmp_path) / f"src_{i}.fit") for i in range(2, 4)
    ]


# ---------------------------------------------------------------------------
# 3. refusal matrix (wrong size/mtime, symlink, rename, wrong subdir, dup)
# ---------------------------------------------------------------------------


def _tamper_size(tmp_path):
    _move_to_stacked(tmp_path, "src_0.fit")
    p = Path(tmp_path) / "stacked" / "src_0.fit"
    p.write_bytes(p.read_bytes() + b"-tampered")


def _tamper_mtime(tmp_path):
    _move_to_stacked(tmp_path, "src_0.fit")
    p = Path(tmp_path) / "stacked" / "src_0.fit"
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def _symlink_destination(tmp_path):
    _move_to_stacked(tmp_path, "src_0.fit")
    p = Path(tmp_path) / "stacked" / "src_0.fit"
    p.unlink()
    try:
        p.symlink_to(Path(tmp_path) / "src_1.fit")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")


def _arbitrary_rename(tmp_path):
    _move_to_stacked(tmp_path, "src_0.fit", dup=None)
    os.rename(
        Path(tmp_path) / "stacked" / "src_0.fit",
        Path(tmp_path) / "stacked" / "src_0_renamed.fit",
    )


def _wrong_stacked_subdir(tmp_path):
    _move_to_stacked(tmp_path, "src_0.fit", subdir="archive")


def _dup_collision(tmp_path):
    _move_to_stacked(tmp_path, "src_0.fit", dup=1234567890)


REFUSALS = [
    ("wrong_size", _tamper_size),
    ("wrong_mtime", _tamper_mtime),
    ("symlink_destination", _symlink_destination),
    ("arbitrary_rename", _arbitrary_rename),
    ("wrong_stacked_subdir", _wrong_stacked_subdir),
    ("dup_collision", _dup_collision),
]


@pytest.mark.parametrize("case", REFUSALS, ids=[c[0] for c in REFUSALS])
def test_safe_resolver_refusals_byte_identical(case, tmp_path):
    _name, corrupt = case
    _write_checkpoint(tmp_path, n_sources=4, frame_count=2)
    corrupt(tmp_path)
    corrupted = _tree_snapshot(tmp_path)

    with pytest.raises(DrizzleCheckpointError):
        read_drizzle_checkpoint(
            str(tmp_path), resolver=SafeStackedSourceResolver("stacked")
        )

    # Rejection never mutates the checkpoint tree / sources.
    assert _tree_snapshot(tmp_path) == corrupted


def test_ambiguous_duplicate_destination_refused(tmp_path):
    """Two distinct plan identities resolving to one path is refused."""
    out_wcs, out_shape_hw = build_output_grid(
        make_wcs(OUT_SHAPE), OUT_SHAPE, 1.0
    )
    cfg = build_drizzle_canonical_config(_fake_qm("square"), product_version="8.2.0")
    writer = DrizzleCheckpointWriter(
        str(tmp_path), "8.2.0", cfg, out_wcs, out_shape_hw
    )
    accs = _accs(out_shape_hw, kernel="square")
    for f in build_frames()[:2]:
        _add_frame_to_all(accs, f)

    ref = Path(tmp_path) / "reference.fit"
    ref.write_bytes(b"reference-bytes")

    # Two byte-identical sources with identical mtime -> identical size+mtime.
    src0 = Path(tmp_path) / "src_0.fit"
    src1 = Path(tmp_path) / "src_1.fit"
    src0.write_bytes(b"twin-source")
    src1.write_bytes(b"twin-source")
    st0 = os.stat(src0)
    os.utime(src0, ns=(st0.st_atime_ns, st0.st_mtime_ns))
    os.utime(src1, ns=(st0.st_atime_ns, st0.st_mtime_ns))
    id0 = _identity(src0)
    id1 = _identity(src1)
    assert id0["size"] == id1["size"] and id0["mtime_ns"] == id1["mtime_ns"]

    binding = {
        "input_roots": [str(tmp_path)],
        "reference": _identity(ref),
        "plan": {"sources": [id0, id1], "decomposition": [2]},
    }
    assert writer.commit(
        accs,
        session_binding=binding,
        counters=_counters(2),
        completed_sources=[id0, id1],
    ) == 1
    before = _tree_snapshot(tmp_path)

    target = id0["path"]

    def bad_resolver(ident, context):
        # Map BOTH plan sources to src_0's path -> ambiguous duplicate.
        if context.get("role") == "plan":
            return [target]
        return [ident["path"]]

    with pytest.raises(DrizzleCheckpointError) as exc_info:
        read_drizzle_checkpoint(str(tmp_path), resolver=bad_resolver)
    assert "ambiguous" in str(exc_info.value)
    assert _tree_snapshot(tmp_path) == before


def test_resolver_invalid_return_refused(tmp_path):
    _write_checkpoint(tmp_path, n_sources=4, frame_count=2)
    before = _tree_snapshot(tmp_path)

    def bad_resolver(ident, context):
        return 42  # not a path / list / None

    with pytest.raises(DrizzleCheckpointError) as exc_info:
        read_drizzle_checkpoint(str(tmp_path), resolver=bad_resolver)
    assert "resolver" in str(exc_info.value)
    assert _tree_snapshot(tmp_path) == before


# ---------------------------------------------------------------------------
# 3b. reference == plan source (legitimate repeated canonical identity)
# ---------------------------------------------------------------------------


def test_strict_default_accepts_reference_also_in_plan(tmp_path):
    """Strict default: reference == plan source 0 is legitimate, not ambiguous."""
    ctx = _write_checkpoint_reference_in_plan(
        tmp_path, n_sources=4, frame_count=2
    )
    before = _tree_snapshot(tmp_path)

    res = read_drizzle_checkpoint(str(tmp_path))

    # Read is read-only and no ambiguity error is raised for the shared identity.
    assert _tree_snapshot(tmp_path) == before
    src0 = str(Path(tmp_path) / "src_0.fit")
    assert res.resolved_reference == src0
    assert res.resolved_reference == res.resolved_plan_paths[0]
    assert list(res.resolved_plan_paths) == [
        str(Path(tmp_path) / f"src_{i}.fit") for i in range(4)
    ]
    assert list(res.resolved_completed_paths) == [
        str(Path(tmp_path) / f"src_{i}.fit") for i in range(2)
    ]
    assert list(res.resolved_remaining_paths) == [
        str(Path(tmp_path) / f"src_{i}.fit") for i in range(2, 4)
    ]
    # Persisted identities stay canonical.
    assert res.session["reference"] == ctx["src_idents"][0]
    assert res.session["plan"]["sources"] == ctx["src_idents"]


def test_safe_resolver_accepts_reference_in_plan_moved_to_stacked(tmp_path):
    """Safe resolver: reference == plan source 0 moved to stacked still resolves."""
    ctx = _write_checkpoint_reference_in_plan(
        tmp_path, n_sources=4, frame_count=2
    )
    _move_to_stacked(tmp_path, "src_0.fit")
    before = _tree_snapshot(tmp_path)

    res = read_drizzle_checkpoint(
        str(tmp_path), resolver=SafeStackedSourceResolver("stacked")
    )

    assert _tree_snapshot(tmp_path) == before
    stacked0 = str(Path(tmp_path) / "stacked" / "src_0.fit")
    assert res.resolved_reference == stacked0
    expected_plan = [stacked0] + [
        str(Path(tmp_path) / f"src_{i}.fit") for i in range(1, 4)
    ]
    assert list(res.resolved_plan_paths) == expected_plan
    assert res.resolved_reference == res.resolved_plan_paths[0]
    assert list(res.resolved_completed_paths) == expected_plan[:2]
    assert list(res.resolved_remaining_paths) == expected_plan[2:]
    assert res.next_source_index == 2
    assert isinstance(res.resolution_policy, SafeStackedSourceResolver)
    # Persisted identities remain canonical (original paths, never stacked).
    assert res.session["reference"] == ctx["src_idents"][0]
    assert res.session["plan"]["sources"] == ctx["src_idents"]


def test_rearm_accepts_reference_in_plan_moved_to_stacked(tmp_path):
    """Re-arm reproduces the safe policy when reference == plan source 0 moved."""
    ctx = _write_checkpoint_reference_in_plan(
        tmp_path, n_sources=6, frame_count=2
    )
    _move_to_stacked(tmp_path, "src_0.fit")

    resolver = SafeStackedSourceResolver("stacked")
    res = read_drizzle_checkpoint(str(tmp_path), resolver=resolver)
    assert res.resolved_reference == str(
        Path(tmp_path) / "stacked" / "src_0.fit"
    )

    cont = DrizzleCheckpointWriter.from_validated_result(res)
    assert cont.generation == 1
    assert cont.next_source_index == 2
    # Continuation state uses canonical (original) identities, never stacked.
    assert cont.session["reference"] == ctx["src_idents"][0]
    assert cont.session["plan"]["sources"] == ctx["src_idents"]
    assert cont.completed_sources == ctx["src_idents"][:2]

    idents = list(cont.session["plan"]["sources"])
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    assert cont.writer.commit(
        cont.accumulators,
        session_binding=_continuation_binding(cont),
        counters=_counters(4),
        completed_sources=idents[:4],
    ) == 2

    m = _load_manifest(tmp_path)
    assert m["generation"] == 2
    assert m["session"]["reference"] == ctx["src_idents"][0]
    assert m["session"]["plan"]["sources"] == ctx["src_idents"]
    assert m["completed_sources"] == ctx["src_idents"][:4]


# ---------------------------------------------------------------------------
# 4. rearm reproduces the immutable policy; never trusts mutable callbacks
# ---------------------------------------------------------------------------


def _continuation_binding(cont):
    return {
        "input_roots": cont.session["input_roots"],
        "reference": cont.session["reference"],
        "plan": cont.session["plan"],
    }


def test_rearm_reproduces_immutable_safe_policy(tmp_path):
    ctx = _write_checkpoint(tmp_path, n_sources=6, frame_count=2)
    _move_to_stacked(tmp_path, "src_0.fit")
    _move_to_stacked(tmp_path, "src_1.fit")

    resolver = SafeStackedSourceResolver("stacked")
    res = read_drizzle_checkpoint(str(tmp_path), resolver=resolver)
    assert isinstance(res.resolution_policy, SafeStackedSourceResolver)
    assert res.resolution_policy is resolver

    # Re-arm: fresh re-read reproduces the carried policy (moved sources still
    # resolve) with no caller-supplied callback.
    cont = DrizzleCheckpointWriter.from_validated_result(res)
    assert isinstance(cont, DrizzleContinuation)
    assert cont.generation == 1
    assert cont.next_source_index == 2
    # Continuation state uses canonical (original) identities, never stacked.
    assert cont.completed_sources == ctx["src_idents"][:2]
    assert cont.session["plan"]["sources"] == ctx["src_idents"]

    # Continue and commit generation 2; the manifest keeps canonical paths.
    idents = list(cont.session["plan"]["sources"])
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    assert cont.writer.commit(
        cont.accumulators,
        session_binding=_continuation_binding(cont),
        counters=_counters(4),
        completed_sources=idents[:4],
    ) == 2

    m = _load_manifest(tmp_path)
    assert m["generation"] == 2
    assert m["completed_sources"] == ctx["src_idents"][:4]
    assert m["session"]["plan"]["sources"] == ctx["src_idents"]


def test_rearm_drops_mutable_callback_and_fails_closed(tmp_path):
    _write_checkpoint(tmp_path, n_sources=4, frame_count=2)
    _move_to_stacked(tmp_path, "src_0.fit")

    mutable = {"subdir": "stacked"}

    def mutable_resolver(ident, context):
        base = os.path.basename(ident["path"])
        return [
            ident["path"],
            os.path.join(os.path.dirname(ident["path"]), mutable["subdir"], base),
        ]

    res = read_drizzle_checkpoint(str(tmp_path), resolver=mutable_resolver)
    # A mutable callable is honoured now but NOT carried as provenance.
    assert res.resolution_policy is None
    assert list(res.resolved_completed_paths) == [
        str(Path(tmp_path) / "stacked" / "src_0.fit"),
        str(Path(tmp_path) / "src_1.fit"),
    ]

    # Re-arm falls back to strict -> the moved source now fails (fail closed).
    with pytest.raises(DrizzleCheckpointError):
        DrizzleCheckpointWriter.from_validated_result(res)


def test_rearm_never_carries_mutable_subclass_policy(tmp_path):
    """A mutable/overriding ``SafeStackedSourceResolver`` subclass is honoured
    for the immediate read but never carried as provenance; re-arm after a
    moved source falls back to strict and fails closed."""
    _write_checkpoint(tmp_path, n_sources=4, frame_count=2)
    _move_to_stacked(tmp_path, "src_0.fit")

    class MutableSubclass(SafeStackedSourceResolver):
        """Subclass smuggling mutable state and overriding resolution."""

        def __post_init__(self):
            super().__post_init__()
            object.__setattr__(self, "resolve_calls", [])

        def resolve(self, ident, context):
            self.resolve_calls.append(ident["path"])
            return super().resolve(ident, context)

    resolver = MutableSubclass("stacked")
    # It really is a subclass (would have passed the old isinstance check) but
    # is not the exact shipped type.
    assert isinstance(resolver, SafeStackedSourceResolver)
    assert type(resolver) is not SafeStackedSourceResolver

    res = read_drizzle_checkpoint(str(tmp_path), resolver=resolver)

    # Immediate read honours the custom resolver (the moved source resolves).
    assert list(res.resolved_completed_paths) == [
        str(Path(tmp_path) / "stacked" / "src_0.fit"),
        str(Path(tmp_path) / "src_1.fit"),
    ]
    assert resolver.resolve_calls  # the custom object was genuinely invoked

    # The custom mutable object is NOT carried as re-arm provenance.
    assert res.resolution_policy is None

    # Re-arm falls back to strict -> the moved source now fails closed.
    with pytest.raises(DrizzleCheckpointError):
        DrizzleCheckpointWriter.from_validated_result(res)


def test_rearm_never_drops_source_validation(tmp_path):
    _write_checkpoint(tmp_path, n_sources=4, frame_count=2)
    _move_to_stacked(tmp_path, "src_0.fit")

    resolver = SafeStackedSourceResolver("stacked")
    res = read_drizzle_checkpoint(str(tmp_path), resolver=resolver)

    # Tamper the stacked destination AFTER the read: re-arm's fresh re-read
    # must still validate and refuse (never trust the prior read).
    stacked0 = Path(tmp_path) / "stacked" / "src_0.fit"
    stacked0.write_bytes(stacked0.read_bytes() + b"-tampered")

    with pytest.raises(DrizzleCheckpointError):
        DrizzleCheckpointWriter.from_validated_result(res)


# ---------------------------------------------------------------------------
# 5. square / lanczos2 continuation remains bit-exact across a move + re-arm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["square", "lanczos2"])
def test_continuation_bit_exact_with_moved_sources(kernel, tmp_path):
    frames = build_frames()
    n_sources = 6

    cont_ref = _accs(OUT_SHAPE, kernel=kernel)
    for f in frames:
        _add_frame_to_all(cont_ref, f)
    cont_img = [a._out_img.copy() for a in cont_ref]
    cont_wht = [a._out_wht.copy() for a in cont_ref]

    _write_checkpoint(tmp_path, kernel=kernel, n_sources=n_sources, frame_count=2)
    _move_to_stacked(tmp_path, "src_0.fit")
    _move_to_stacked(tmp_path, "src_1.fit")

    resolver = SafeStackedSourceResolver("stacked")
    res = read_drizzle_checkpoint(str(tmp_path), resolver=resolver)
    assert res.next_source_index == 2

    c1 = DrizzleCheckpointWriter.from_validated_result(res)
    idents = list(c1.session["plan"]["sources"])
    for f in frames[2:4]:
        _add_frame_to_all(c1.accumulators, f)
    assert c1.writer.commit(
        c1.accumulators,
        session_binding=_continuation_binding(c1),
        counters=_counters(4),
        completed_sources=idents[:4],
    ) == 2

    res2 = read_drizzle_checkpoint(str(tmp_path), resolver=resolver)
    assert res2.generation == 2
    assert res2.next_source_index == 4
    ref4 = _accs(OUT_SHAPE, kernel=kernel)
    for f in frames[:4]:
        _add_frame_to_all(ref4, f)
    for c in range(3):
        assert np.array_equal(res2.accumulators[c]._out_img, ref4[c]._out_img)
        assert np.array_equal(res2.accumulators[c]._out_wht, ref4[c]._out_wht)

    # Continue to the end and compare against the uninterrupted reference.
    c2 = DrizzleCheckpointWriter.from_validated_result(res2)
    for f in frames[4:6]:
        _add_frame_to_all(c2.accumulators, f)
    assert c2.writer.commit(
        c2.accumulators,
        session_binding=_continuation_binding(c2),
        counters=_counters(6),
        completed_sources=idents[:6],
    ) == 3

    res3 = read_drizzle_checkpoint(str(tmp_path), resolver=resolver)
    assert res3.generation == 3
    assert res3.next_source_index == 6
    for c in range(3):
        assert np.array_equal(res3.accumulators[c]._out_img, cont_img[c])
        assert np.array_equal(res3.accumulators[c]._out_wht, cont_wht[c])

    if kernel == "lanczos2":
        assert np.any(cont_wht[0] < 0.0)
        assert np.array_equal(res3.accumulators[0]._out_wht, cont_wht[0])


# ---------------------------------------------------------------------------
# 6. resolver policy object contract (immutable, invalid subdir refused)
# ---------------------------------------------------------------------------


def test_safe_resolver_rejects_invalid_subdir():
    with pytest.raises(DrizzleCheckpointError):
        SafeStackedSourceResolver("")
    with pytest.raises(DrizzleCheckpointError):
        SafeStackedSourceResolver("../etc")
    with pytest.raises(DrizzleCheckpointError):
        SafeStackedSourceResolver("/abs")
    with pytest.raises(DrizzleCheckpointError):
        SafeStackedSourceResolver("a/b")


def test_safe_resolver_is_immutable():
    r = SafeStackedSourceResolver("stacked")
    with pytest.raises(Exception):
        r.stacked_subdir_name = "other"
    # A _dup_ collision basename is never guessed.
    ident = {"path": os.path.join("d", "src_0_dup_123.fit")}
    assert r.resolve(ident, {}) is None
    ident2 = {"path": os.path.join("d", "src_0.fit")}
    assert r.resolve(ident2, {}) == [
        os.path.join("d", "src_0.fit"),
        os.path.join("d", "stacked", "src_0.fit"),
    ]
