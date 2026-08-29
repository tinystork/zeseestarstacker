"""RSM2-D2B1: continuation-writer seam (re-arm N -> N+1) for native Drizzle.

Proves the narrow, safe continuation seam added after the D2A reader:
:meth:`DrizzleCheckpointWriter.from_validated_result` re-arms the atomic writer
at generation ``N+1`` **only** from an already-validated
:class:`DrizzleCheckpointResult`, preserving the D1 fresh-run namespace refusal
and the D1 atomic commit protocol, and enforcing monotonic continuation (no
rollback / rewrite / reorder / divergent prefix and no cumulative-counter or
native per-channel exposure rollback).

The factory returns a dedicated :class:`DrizzleContinuation` re-arm result that
carries the **fresh** writer plus the **fresh** reconstructed accumulators /
session / counters / ledger (re-read from disk), so the lifecycle cannot
accidentally continue from stale/tampered result payloads.

Covered:

A. fresh-run namespace refusal regression + no public ``allow_existing`` bypass;
B. re-arm is a no-write / no-GC operation, returns a fresh
   :class:`DrizzleContinuation`, refuses arbitrary dicts / stale provenance /
   a tampered generation token, and **ignores** mutated shallow-frozen payloads
   of the originally supplied result (manifest/session/counters/config/WCS/
   accumulator list);
C. write -> read -> re-arm -> continue -> commit -> read cycles (``N+1`` and
   ``N+1 -> N+2``) with bit-exact native buffers, exact next index / counters,
   for ``square`` and *signed-WHT* ``lanczos2``;
D. rollback / reorder / divergent counter / ledger refusals before any write,
   including cumulative-truth counters (``exposure_unknown_count``,
   ``exposure_min``/``exposure_max``) and native per-channel ``total_exptime``,
   plus the same-writer two-commit monotonic-baseline regression;
E. failure-injection matrix (array / config / manifest temp / replace) leaving
   generation N byte-identical with no created attempt files;
F. two continuation writers racing from the same result: fail-closed, no
   corruption of the committed generation.
"""

import dataclasses
import hashlib
import json
import math
import os
import threading
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

from seestar.core.drizzle_checkpoint import (
    CHECKPOINT_DIRNAME,
    MANIFEST_FILENAME,
    RUN_CONFIG_FILENAME,
    DrizzleCheckpointError,
    DrizzleCheckpointResult,
    DrizzleCheckpointWriter,
    DrizzleContinuation,
    build_drizzle_canonical_config,
    read_drizzle_checkpoint,
)
from seestar.core.drizzle_core import (
    DrizzleAccumulator,
    build_output_grid,
)
from seestar import run_contract


# ---------------------------------------------------------------------------
# deterministic synthetic frames / WCS / identities
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


def _writer(tmp_path, kernel="square"):
    wcs = make_wcs(OUT_SHAPE)
    out_wcs, out_shape_hw = build_output_grid(wcs, OUT_SHAPE, 1.0)
    cfg = build_drizzle_canonical_config(_fake_qm(kernel), product_version="8.2.0")
    return (
        DrizzleCheckpointWriter(
            str(tmp_path), "8.2.0", cfg, out_wcs, out_shape_hw
        ),
        out_wcs,
        out_shape_hw,
    )


def _accs(shape, kernel="square"):
    return [
        DrizzleAccumulator(shape, kernel=kernel, pixfrac=1.0) for _ in range(3)
    ]


def _identity(path):
    st = os.stat(path)
    return {
        "path": os.path.normcase(str(path)),
        "name": os.path.basename(str(path)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _identities(tmp_path, n):
    idents = []
    for i in range(n):
        p = Path(tmp_path) / f"src_{i}.fit"
        p.write_bytes(b"src-data-%d" % i)
        idents.append(_identity(p))
    return idents


def _plan(sources):
    return {"sources": sources, "decomposition": [len(sources)]}


def _binding(tmp_path, plan):
    ref = Path(tmp_path) / "reference.fit"
    ref.write_bytes(b"reference-bytes")
    return {
        "input_roots": [str(tmp_path)],
        "reference": _identity(ref),
        "plan": plan,
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


def _gen1_snapshot(tmp_path):
    """Snapshot the committed generation-1 manifest bytes + every artifact."""
    ckpt = _ckpt(tmp_path)
    manifest_bytes = (ckpt / MANIFEST_FILENAME).read_bytes()
    artifacts = {
        n: (ckpt / n).read_bytes()
        for n in sorted(os.listdir(ckpt))
        if n.endswith(".npy")
    }
    cfg_bytes = (Path(tmp_path) / RUN_CONFIG_FILENAME).read_bytes()
    return manifest_bytes, artifacts, cfg_bytes


def _assert_gen1_intact(tmp_path, manifest_bytes, artifacts, cfg_bytes):
    ckpt = _ckpt(tmp_path)
    assert (ckpt / MANIFEST_FILENAME).read_bytes() == manifest_bytes
    for name, data in artifacts.items():
        assert (ckpt / name).exists(), f"{name} missing after failure"
        assert (ckpt / name).read_bytes() == data
    assert (Path(tmp_path) / RUN_CONFIG_FILENAME).read_bytes() == cfg_bytes


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


def _commit_gen1(tmp_path, kernel="square", n_sources=6, frame_count=2,
                 counters_overrides=None):
    writer, _wcs, out_shape_hw = _writer(tmp_path, kernel=kernel)
    accs = _accs(out_shape_hw, kernel=kernel)
    frames = build_frames()
    for f in frames[:frame_count]:
        _add_frame_to_all(accs, f)
    idents = _identities(tmp_path, n_sources)
    binding = _binding(tmp_path, _plan(idents))
    counters = _counters(frame_count)
    if counters_overrides:
        counters.update(counters_overrides)
    gen = writer.commit(
        accs,
        session_binding=binding,
        counters=counters,
        completed_sources=idents[:frame_count],
    )
    assert gen == 1
    return writer, accs, frames, idents, binding


def _read_and_rearm(tmp_path):
    """Fresh read + re-arm; returns (continuation, idents, binding)."""
    res = read_drizzle_checkpoint(str(tmp_path))
    cont = DrizzleCheckpointWriter.from_validated_result(res)
    idents = list(cont.session["plan"]["sources"])
    binding = {
        "input_roots": cont.session["input_roots"],
        "reference": cont.session["reference"],
        "plan": cont.session["plan"],
    }
    return cont, idents, binding


def _rearm_gen1(tmp_path, kernel="square"):
    """Commit gen1, re-arm, and continue with frames 2-3 (in memory)."""
    _commit_gen1(tmp_path, kernel=kernel, n_sources=6, frame_count=2)
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    return cont, idents, binding


def _assert_gen1_still_authoritative(tmp_path):
    manifest = _load_manifest(tmp_path)
    assert manifest["generation"] == 1
    assert not any("gen-00000002" in n for n in os.listdir(_ckpt(tmp_path)))


# ---------------------------------------------------------------------------
# A. fresh-run namespace refusal regression + no public bypass
# ---------------------------------------------------------------------------


def test_fresh_writer_still_refuses_nonempty_checkpoint(tmp_path):
    _commit_gen1(tmp_path, kernel="square", n_sources=4, frame_count=2)
    before = _tree_snapshot(tmp_path)

    # A freshly constructed writer on the same output must still refuse at
    # construction, exactly as D1 (the continuation seam must not weaken this).
    with pytest.raises(DrizzleCheckpointError):
        _writer(tmp_path)

    assert _tree_snapshot(tmp_path) == before


def test_no_public_allow_existing_bypass(tmp_path):
    wcs = make_wcs(OUT_SHAPE)
    out_wcs, out_shape_hw = build_output_grid(wcs, OUT_SHAPE, 1.0)
    cfg = build_drizzle_canonical_config(_fake_qm("square"), product_version="8.2.0")
    with pytest.raises(TypeError):
        DrizzleCheckpointWriter(
            str(tmp_path), "8.2.0", cfg, out_wcs, out_shape_hw,
            allow_existing=True,
        )
    # No checkpoint directory was created by the rejected construction.
    assert not _ckpt(tmp_path).exists()


# ---------------------------------------------------------------------------
# B. re-arm is no-write / no-GC; rejects dicts and stale provenance; ignores
#    tampered result payloads
# ---------------------------------------------------------------------------


def test_rearm_rejects_arbitrary_dict(tmp_path):
    with pytest.raises(DrizzleCheckpointError):
        DrizzleCheckpointWriter.from_validated_result(
            {"generation": 1, "source_output_dir": str(tmp_path)}
        )


def test_rearm_is_nowrite_and_nogc(tmp_path):
    _commit_gen1(tmp_path, kernel="square", n_sources=6, frame_count=2)
    before = _tree_snapshot(tmp_path)

    res = read_drizzle_checkpoint(str(tmp_path))
    cont = DrizzleCheckpointWriter.from_validated_result(res)

    # The factory returns a dedicated re-arm result, not a bare writer.
    assert isinstance(cont, DrizzleContinuation)
    cw = cont.writer
    assert isinstance(cw, DrizzleCheckpointWriter)

    # Provenance is immutable, normalized, and the writer binds to it.
    assert isinstance(res, DrizzleCheckpointResult)
    assert res.source_output_dir == os.path.realpath(str(tmp_path))
    assert cw.output_dir == res.source_output_dir
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.source_output_dir = "/elsewhere"

    # Initial state: current_generation == N, next_generation == N + 1.
    assert res.generation == 1
    assert cont.generation == 1
    assert cont.next_source_index == 2
    assert cw.current_generation == 1
    assert cw.next_generation == 2

    # The continuation carries fresh accumulators (three, bit-equal to disk).
    assert len(cont.accumulators) == 3
    assert cont.session["input_roots"] == [str(tmp_path)]
    assert cont.counters["frame_count"] == 2
    assert cont.completed_sources == list(res.session["plan"]["sources"])[:2]

    # Re-arm performed no writes and no GC: tree byte-identical, gen-1 intact.
    assert _tree_snapshot(tmp_path) == before
    assert any("gen-00000001" in n for n in os.listdir(_ckpt(tmp_path)))


def test_rearm_refuses_stale_result_after_another_commit(tmp_path):
    _commit_gen1(tmp_path, kernel="square", n_sources=6, frame_count=2)
    res1 = read_drizzle_checkpoint(str(tmp_path))

    # First continuation commits generation 2 from res1.
    cont1 = DrizzleCheckpointWriter.from_validated_result(res1)
    idents = list(cont1.session["plan"]["sources"])
    binding = {
        "input_roots": cont1.session["input_roots"],
        "reference": cont1.session["reference"],
        "plan": cont1.session["plan"],
    }
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont1.accumulators, f)
    gen = cont1.writer.commit(
        cont1.accumulators,
        session_binding=binding,
        counters=_counters(4),
        completed_sources=idents[:4],
    )
    assert gen == 2

    # Re-arming again from the STALE res1 (generation 1) must fail closed.
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        DrizzleCheckpointWriter.from_validated_result(res1)
    assert "stale" in str(exc_info.value)


def test_rearm_rejects_tampered_generation_token(tmp_path):
    _commit_gen1(tmp_path, kernel="square", n_sources=6, frame_count=2)
    res = read_drizzle_checkpoint(str(tmp_path))
    # Bypass the frozen field to tamper the stale-result token.
    object.__setattr__(res, "generation", 99)
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        DrizzleCheckpointWriter.from_validated_result(res)
    assert "stale" in str(exc_info.value)


def test_rearm_ignores_mutated_result_payloads(tmp_path):
    """Tampered shallow-frozen payloads of the supplied result cannot influence
    re-arm: the factory fresh-reloads and binds to untampered disk truth."""
    _commit_gen1(tmp_path, kernel="square", n_sources=6, frame_count=2)
    res = read_drizzle_checkpoint(str(tmp_path))

    # Mutate every shallow-frozen (but still mutable) payload of the result.
    res.manifest["generation"] = 999
    res.manifest["frame_count"] = 999
    res.session["input_roots"] = ["/tampered/root"]
    res.session["plan"]["sources"] = []
    res.counters["frame_count"] = 999
    res.counters["total_exposure_seconds"] = 0.0
    res.completed_sources.clear()
    res.accumulators.clear()
    res.config.scientific["drizzle_kernel_effective"] = "gaussian"
    res.wcs.array_shape = (1, 1)

    # Re-arm must still bind to fresh, untampered disk truth.
    cont = DrizzleCheckpointWriter.from_validated_result(res)
    assert cont.generation == 1
    assert cont.next_source_index == 2
    assert cont.counters["frame_count"] == 2
    assert cont.counters["total_exposure_seconds"] == sum(FRAME_EXPTIMES[:2])
    assert cont.session["input_roots"] == [str(tmp_path)]
    assert len(cont.completed_sources) == 2
    assert len(cont.accumulators) == 3
    # The writer bound the canonical identity from the fresh config/WCS.
    assert cont.writer.fingerprint is not None
    assert cont.writer.output_shape_hw == OUT_SHAPE


# ---------------------------------------------------------------------------
# C. write -> read -> re-arm -> continue -> commit -> read cycles (bit-exact)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["square", "lanczos2"])
def test_full_continuation_cycle_n1_and_n2(kernel, tmp_path):
    frames = build_frames()
    n_sources = 6

    # Continuous reference (all six frames).
    cont_ref = _accs(OUT_SHAPE, kernel=kernel)
    for f in frames:
        _add_frame_to_all(cont_ref, f)
    cont_img = [a._out_img.copy() for a in cont_ref]
    cont_wht = [a._out_wht.copy() for a in cont_ref]

    # Generation 1 (frames 0,1).
    _commit_gen1(tmp_path, kernel=kernel, n_sources=n_sources, frame_count=2)
    res1 = read_drizzle_checkpoint(str(tmp_path))
    assert res1.generation == 1
    assert res1.next_source_index == 2

    # Re-arm -> continue (frames 2,3) -> commit generation 2.
    c1 = DrizzleCheckpointWriter.from_validated_result(res1)
    assert c1.writer.current_generation == 1 and c1.writer.next_generation == 2
    assert c1.generation == 1 and c1.next_source_index == 2
    idents = list(c1.session["plan"]["sources"])
    binding = {
        "input_roots": c1.session["input_roots"],
        "reference": c1.session["reference"],
        "plan": c1.session["plan"],
    }
    for f in frames[2:4]:
        _add_frame_to_all(c1.accumulators, f)
    assert c1.writer.commit(
        c1.accumulators,
        session_binding=binding,
        counters=_counters(4),
        completed_sources=idents[:4],
    ) == 2

    res2 = read_drizzle_checkpoint(str(tmp_path))
    assert res2.generation == 2
    assert res2.next_source_index == 4
    assert res2.counters["frame_count"] == 4
    assert res2.counters["total_exposure_seconds"] == sum(FRAME_EXPTIMES[:4])
    # Generation 2 must equal a continuous run that stopped at 4 frames.
    ref4 = _accs(OUT_SHAPE, kernel=kernel)
    for f in frames[:4]:
        _add_frame_to_all(ref4, f)
    for c in range(3):
        assert np.array_equal(res2.accumulators[c]._out_img, ref4[c]._out_img)
        assert np.array_equal(res2.accumulators[c]._out_wht, ref4[c]._out_wht)

    # Re-arm again -> continue (frames 4,5) -> commit generation 3.
    c2 = DrizzleCheckpointWriter.from_validated_result(res2)
    assert c2.writer.current_generation == 2 and c2.writer.next_generation == 3
    for f in frames[4:6]:
        _add_frame_to_all(c2.accumulators, f)
    assert c2.writer.commit(
        c2.accumulators,
        session_binding=binding,
        counters=_counters(6),
        completed_sources=idents[:6],
    ) == 3

    res3 = read_drizzle_checkpoint(str(tmp_path))
    assert res3.generation == 3
    assert res3.next_source_index == 6
    for c in range(3):
        assert np.array_equal(res3.accumulators[c]._out_img, cont_img[c])
        assert np.array_equal(res3.accumulators[c]._out_wht, cont_wht[c])

    if kernel == "lanczos2":
        # The signed native WHT survived the two continuation commits
        # bit-exactly, including the negative lobes.
        assert np.any(cont_wht[0] < 0.0)
        assert np.array_equal(res3.accumulators[0]._out_wht, cont_wht[0])


def test_continuation_manifest_authoritative_and_gc(tmp_path):
    _commit_gen1(tmp_path, kernel="square", n_sources=6, frame_count=2)
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    assert cont.writer.commit(
        cont.accumulators, session_binding=binding, counters=_counters(4),
        completed_sources=idents[:4],
    ) == 2

    manifest = _load_manifest(tmp_path)
    assert manifest["generation"] == 2
    gen2_names = {
        d["file"] for ch in manifest["channels"] for d in (ch["out_img"], ch["out_wht"])
    }
    assert all("gen-00000002" in n for n in gen2_names)
    # Generation 1 was GC'd only AFTER the generation-2 commit.
    assert not any("gen-00000001" in n for n in os.listdir(_ckpt(tmp_path)))
    # Generation 2 artifacts all present.
    for n in gen2_names:
        assert (_ckpt(tmp_path) / n).exists()


# ---------------------------------------------------------------------------
# D. rollback / reorder / divergent counter / ledger refusals before writes
# ---------------------------------------------------------------------------


def test_continuation_rejects_rollback_frame_count(tmp_path):
    cont, idents, binding = _rearm_gen1(tmp_path)
    before = _tree_snapshot(tmp_path)
    # frame_count 2 == loaded frame_count (no extension / rollback).
    with pytest.raises(DrizzleCheckpointError):
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=_counters(2),
            completed_sources=idents[:2],
        )
    _assert_gen1_still_authoritative(tmp_path)
    assert _tree_snapshot(tmp_path) == before


def test_continuation_rejects_rollback_exposure(tmp_path):
    cont, idents, binding = _rearm_gen1(tmp_path)
    # frame_count extends to 4 but total exposure rolled back below loaded 2.1.
    counters = _counters(4)
    counters["total_exposure_seconds"] = 1.0
    with pytest.raises(DrizzleCheckpointError):
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=counters,
            completed_sources=idents[:4],
        )
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_rejects_reordered_plan(tmp_path):
    cont, idents, binding = _rearm_gen1(tmp_path)
    reordered = {
        "input_roots": binding["input_roots"],
        "reference": binding["reference"],
        "plan": {"sources": [idents[1], idents[0]] + idents[2:],
                 "decomposition": [len(idents)]},
    }
    with pytest.raises(DrizzleCheckpointError):
        cont.writer.commit(
            cont.accumulators, session_binding=reordered, counters=_counters(4),
            completed_sources=reordered["plan"]["sources"][:4],
        )
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_rejects_divergent_ledger_prefix(tmp_path):
    cont, idents, binding = _rearm_gen1(tmp_path)
    # Rewrite the already-committed ledger prefix (swap first two entries).
    rewritten = [idents[1], idents[0]] + idents[2:4]
    with pytest.raises(DrizzleCheckpointError):
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=_counters(4),
            completed_sources=rewritten,
        )
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_rejects_divergent_reference(tmp_path):
    cont, idents, binding = _rearm_gen1(tmp_path)
    alt_ref = Path(tmp_path) / "other_reference.fit"
    alt_ref.write_bytes(b"different-reference")
    divergent = {
        "input_roots": binding["input_roots"],
        "reference": _identity(alt_ref),
        "plan": binding["plan"],
    }
    with pytest.raises(DrizzleCheckpointError):
        cont.writer.commit(
            cont.accumulators, session_binding=divergent, counters=_counters(4),
            completed_sources=idents[:4],
        )
    _assert_gen1_still_authoritative(tmp_path)


# -- cumulative-truth counters (D2B1 finding 3) -----------------------------


def test_continuation_rejects_exposure_unknown_count_decrease(tmp_path):
    # Loaded checkpoint has exposure_unknown_count == 1 (known unknown frame).
    _commit_gen1(
        tmp_path, n_sources=6, frame_count=2,
        counters_overrides={"exposure_unknown_count": 1},
    )
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    counters = _counters(4)
    counters["exposure_unknown_count"] = 0  # decreased from loaded 1
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=counters,
            completed_sources=idents[:4],
        )
    assert "unknown_count" in str(exc_info.value)
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_rejects_exposure_min_increase(tmp_path):
    # Loaded checkpoint has known exposure_min == 1.0 / max == 1.1.
    _commit_gen1(
        tmp_path, n_sources=6, frame_count=2,
        counters_overrides={"exposure_min": 1.0, "exposure_max": 1.1},
    )
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    counters = _counters(4)
    counters["exposure_min"] = 1.5  # increased above loaded 1.0
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=counters,
            completed_sources=idents[:4],
        )
    assert "exposure_min" in str(exc_info.value)
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_rejects_exposure_max_decrease(tmp_path):
    _commit_gen1(
        tmp_path, n_sources=6, frame_count=2,
        counters_overrides={"exposure_min": 1.0, "exposure_max": 1.1},
    )
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    counters = _counters(4)
    counters["exposure_max"] = 0.9  # decreased below loaded 1.1
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=counters,
            completed_sources=idents[:4],
        )
    assert "exposure_max" in str(exc_info.value)
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_rejects_exposure_min_disappear(tmp_path):
    _commit_gen1(
        tmp_path, n_sources=6, frame_count=2,
        counters_overrides={"exposure_min": 1.0, "exposure_max": 1.1},
    )
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    counters = _counters(4)
    counters["exposure_min"] = None  # known -> unknown not allowed
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=counters,
            completed_sources=idents[:4],
        )
    assert "exposure_min" in str(exc_info.value)
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_rejects_exposure_max_disappear(tmp_path):
    _commit_gen1(
        tmp_path, n_sources=6, frame_count=2,
        counters_overrides={"exposure_min": 1.0, "exposure_max": 1.1},
    )
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    counters = _counters(4)
    counters["exposure_max"] = None  # known -> unknown not allowed
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=counters,
            completed_sources=idents[:4],
        )
    assert "exposure_max" in str(exc_info.value)
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_allows_unknown_minmax_becoming_known(tmp_path):
    # Loaded checkpoint has unknown min/max (None); later known frames may make
    # them known — that transition is semantically legal and must succeed.
    _commit_gen1(
        tmp_path, n_sources=6, frame_count=2,
        counters_overrides={"exposure_min": None, "exposure_max": None},
    )
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    counters = _counters(4)  # now known min/max
    assert cont.writer.commit(
        cont.accumulators, session_binding=binding, counters=counters,
        completed_sources=idents[:4],
    ) == 2
    manifest = _load_manifest(tmp_path)
    assert manifest["generation"] == 2
    assert manifest["exposure_min"] == min(FRAME_EXPTIMES[:4])
    assert manifest["exposure_max"] == max(FRAME_EXPTIMES[:4])


def test_continuation_rejects_per_channel_total_rollback(tmp_path):
    cont, idents, binding = _rearm_gen1(tmp_path)
    # frame_count extends to 4, but the native per-channel total_exptime is
    # rolled back below the loaded value.
    for acc in cont.accumulators:
        acc._total_exptime = 0.0
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=_counters(4),
            completed_sources=idents[:4],
        )
    assert "total_exptime" in str(exc_info.value)
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_rejects_per_channel_total_divergence(tmp_path):
    cont, idents, binding = _rearm_gen1(tmp_path)
    # frame_count extends to 4 but native per-channel total_exptime is unchanged
    # (divergent: must strictly increase).
    for acc in cont.accumulators:
        acc._total_exptime = sum(FRAME_EXPTIMES[:2])
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=_counters(4),
            completed_sources=idents[:4],
        )
    assert "total_exptime" in str(exc_info.value)
    _assert_gen1_still_authoritative(tmp_path)


def test_same_writer_two_commit_monotonic_baseline(tmp_path):
    """A single continuation writer must advance its monotonic baseline after
    each successful commit: a later rollback attempt at N+2 is refused
    byte-identically, while a valid N+2 extension succeeds."""
    _commit_gen1(tmp_path, kernel="square", n_sources=6, frame_count=2)
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()

    # N+1: commit generation 2 (frames 2,3).
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    assert cont.writer.commit(
        cont.accumulators, session_binding=binding, counters=_counters(4),
        completed_sources=idents[:4],
    ) == 2
    assert cont.writer.current_generation == 2
    assert cont.writer.next_generation == 3

    # N+2 rollback attempt: frame_count back to 3 (and ledger/exposure rolled
    # back relative to N+1) must be refused byte-identically.
    snap = _gen1_snapshot(tmp_path)  # snapshot of committed gen-2 state
    with pytest.raises(DrizzleCheckpointError):
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=_counters(3),
            completed_sources=idents[:3],
        )
    # Nothing changed: committed generation 2 stays byte-identical.
    _assert_gen1_intact(tmp_path, *snap)
    manifest = _load_manifest(tmp_path)
    assert manifest["generation"] == 2

    # Valid N+2 extension (frames 4,5) succeeds on the SAME writer.
    for f in frames[4:6]:
        _add_frame_to_all(cont.accumulators, f)
    assert cont.writer.commit(
        cont.accumulators, session_binding=binding, counters=_counters(6),
        completed_sources=idents[:6],
    ) == 3
    res3 = read_drizzle_checkpoint(str(tmp_path))
    assert res3.generation == 3
    assert res3.next_source_index == 6


# ---------------------------------------------------------------------------
# E. failure-injection matrix preserving generation N byte-identically
# ---------------------------------------------------------------------------


def _rearm_gen1_with_snapshot(tmp_path, kernel="square"):
    _commit_gen1(tmp_path, kernel=kernel, n_sources=6, frame_count=2)
    snap = _gen1_snapshot(tmp_path)
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    return cont, idents, binding, snap


def _attempt_gen2(cont, idents, binding):
    return cont.writer.commit(
        cont.accumulators, session_binding=binding, counters=_counters(4),
        completed_sources=idents[:4],
    )


def test_continuation_failure_at_array_stage_preserves_n(tmp_path, monkeypatch):
    cont, idents, binding, snap = _rearm_gen1_with_snapshot(tmp_path)
    calls = {"n": 0}
    orig = cont.writer._write_array_artifact

    def failing(arr, name):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("injected array failure")
        return orig(arr, name)

    monkeypatch.setattr(cont.writer, "_write_array_artifact", failing)
    with pytest.raises(DrizzleCheckpointError):
        _attempt_gen2(cont, idents, binding)
    _assert_gen1_intact(tmp_path, *snap)
    assert not any("gen-00000002" in n for n in os.listdir(_ckpt(tmp_path)))


def test_continuation_failure_at_cfg_stage_preserves_n(tmp_path, monkeypatch):
    cont, idents, binding, snap = _rearm_gen1_with_snapshot(tmp_path)

    def fail_cfg(*a, **k):
        raise RuntimeError("injected cfg failure")

    monkeypatch.setattr(run_contract, "write_cfg", fail_cfg)
    with pytest.raises(DrizzleCheckpointError):
        _attempt_gen2(cont, idents, binding)
    _assert_gen1_intact(tmp_path, *snap)
    assert not any("gen-00000002" in n for n in os.listdir(_ckpt(tmp_path)))


def test_continuation_failure_at_manifest_write_preserves_n(tmp_path, monkeypatch):
    cont, idents, binding, snap = _rearm_gen1_with_snapshot(tmp_path)

    def fail_manifest(manifest):
        raise RuntimeError("injected manifest write failure")

    monkeypatch.setattr(cont.writer, "_write_manifest", fail_manifest)
    with pytest.raises(DrizzleCheckpointError):
        _attempt_gen2(cont, idents, binding)
    _assert_gen1_intact(tmp_path, *snap)
    assert not any("gen-00000002" in n for n in os.listdir(_ckpt(tmp_path)))


def test_continuation_failure_at_manifest_replace_preserves_n(tmp_path, monkeypatch):
    cont, idents, binding, snap = _rearm_gen1_with_snapshot(tmp_path)
    real_replace = os.replace

    def replace_fail(src, dst):
        if str(dst).endswith(MANIFEST_FILENAME):
            raise RuntimeError("injected replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_fail)
    with pytest.raises(DrizzleCheckpointError):
        _attempt_gen2(cont, idents, binding)
    _assert_gen1_intact(tmp_path, *snap)
    # No created attempt artifacts nor foreign/owned manifest temps remain.
    assert not any("gen-00000002" in n for n in os.listdir(_ckpt(tmp_path)))


# ---------------------------------------------------------------------------
# F. two continuation writers racing from the same result: fail-closed
# ---------------------------------------------------------------------------


def test_two_continuation_writers_race_fail_closed(tmp_path):
    _commit_gen1(tmp_path, kernel="square", n_sources=6, frame_count=2)
    res = read_drizzle_checkpoint(str(tmp_path))
    frames = build_frames()

    # Two re-arm results from the SAME validated result (both bound to gen 2).
    cont_a = DrizzleCheckpointWriter.from_validated_result(res)
    res_b = read_drizzle_checkpoint(str(tmp_path))
    cont_b = DrizzleCheckpointWriter.from_validated_result(res_b)

    idents = list(cont_a.session["plan"]["sources"])
    binding = {
        "input_roots": cont_a.session["input_roots"],
        "reference": cont_a.session["reference"],
        "plan": cont_a.session["plan"],
    }

    # Each thread continues its own fresh reconstructed accumulators (frames 2,3).
    for f in frames[2:4]:
        _add_frame_to_all(cont_a.accumulators, f)
        _add_frame_to_all(cont_b.accumulators, f)

    results = {}

    def run_a():
        try:
            g = cont_a.writer.commit(
                cont_a.accumulators, session_binding=binding,
                counters=_counters(4), completed_sources=idents[:4],
            )
            results["a"] = f"committed:{g}"
        except DrizzleCheckpointError as exc:
            results["a"] = f"refused:{exc}"

    def run_b():
        try:
            g = cont_b.writer.commit(
                cont_b.accumulators, session_binding=binding,
                counters=_counters(4), completed_sources=idents[:4],
            )
            results["b"] = f"committed:{g}"
        except DrizzleCheckpointError as exc:
            results["b"] = f"refused:{exc}"

    ta = threading.Thread(target=run_a)
    tb = threading.Thread(target=run_b)
    ta.start()
    tb.start()
    ta.join(timeout=60)
    tb.join(timeout=60)
    assert not ta.is_alive() and not tb.is_alive()

    # Exactly one continuation committed generation 2; the other failed closed.
    committed = [v for v in results.values() if v.startswith("committed:")]
    refused = [v for v in results.values() if v.startswith("refused:")]
    assert len(committed) == 1 and len(refused) == 1, results
    assert committed[0] == "committed:2"
    assert "already exists" in refused[0], results

    # The committed generation 2 is valid and byte-consistent.
    manifest = _load_manifest(tmp_path)
    assert manifest["generation"] == 2
    ckpt = _ckpt(tmp_path)
    for ch in manifest["channels"]:
        c = ch["channel"]
        for kind in ("out_img", "out_wht"):
            desc = ch[kind]
            p = ckpt / desc["file"]
            assert p.is_file()
            raw = p.read_bytes()
            assert desc["size"] == len(raw)
            assert desc["sha256"] == hashlib.sha256(raw).hexdigest()

    # A fresh read of the committed generation 2 still validates.
    res2 = read_drizzle_checkpoint(str(tmp_path))
    assert res2.generation == 2
    assert res2.next_source_index == 4


# ---------------------------------------------------------------------------
# G. no fallible continuation-state work after the manifest commit (finding 1)
# ---------------------------------------------------------------------------


def test_continuation_next_state_prep_failure_before_writes(tmp_path, monkeypatch):
    """A fallible failure while preparing the next continuation baseline must
    occur before any artifact write and preserve generation N byte-identically."""
    cont, idents, binding, snap = _rearm_gen1_with_snapshot(tmp_path)

    def fail_build(*args, **kwargs):
        raise RuntimeError("injected next-state preparation failure")

    monkeypatch.setattr(cont.writer, "_build_next_continuation_state", fail_build)
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        _attempt_gen2(cont, idents, binding)
    assert "injected next-state preparation failure" in str(exc_info.value)
    _assert_gen1_intact(tmp_path, *snap)
    # The failure happened before any artifact / manifest-temp write.
    assert not any("gen-00000002" in n for n in os.listdir(_ckpt(tmp_path)))


def test_continuation_deepcopy_failure_before_writes(tmp_path, monkeypatch):
    """The deep-copy that materializes the next baseline is fallible and must
    run entirely before any write: a MemoryError there leaves N byte-identical."""
    import copy as copy_module

    cont, idents, binding, snap = _rearm_gen1_with_snapshot(tmp_path)

    def fail_deepcopy(obj, *args, **kwargs):
        raise MemoryError("injected deepcopy failure")

    monkeypatch.setattr(copy_module, "deepcopy", fail_deepcopy)
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        _attempt_gen2(cont, idents, binding)
    assert "injected deepcopy failure" in str(exc_info.value)
    _assert_gen1_intact(tmp_path, *snap)
    assert not any("gen-00000002" in n for n in os.listdir(_ckpt(tmp_path)))


# ---------------------------------------------------------------------------
# H. cumulative unknown/known arithmetic (finding 2)
# ---------------------------------------------------------------------------


def test_continuation_rejects_retroactive_unknown_inflation(tmp_path):
    """The unknown count may only grow by counting *new* frames, never by
    retroactively reclassifying already-committed known frames."""
    _commit_gen1(tmp_path, n_sources=6, frame_count=2)  # unknown == 0
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    counters = _counters(4)
    counters["exposure_unknown_count"] = 3  # delta_unknown 3 > delta_frame 2
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=counters,
            completed_sources=idents[:4],
        )
    assert "retroactive" in str(exc_info.value)
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_rejects_total_fabrication_when_all_new_unknown(tmp_path):
    """When every new frame is unknown, the cumulative total cannot change."""
    _commit_gen1(tmp_path, n_sources=6, frame_count=2)  # known: total 2.1
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    counters = _counters(4)
    counters["exposure_unknown_count"] = 2  # all new frames unknown
    counters["total_exposure_seconds"] = (
        cont.counters["total_exposure_seconds"] + 1.0
    )
    counters["exposure_min"] = cont.counters["exposure_min"]
    counters["exposure_max"] = cont.counters["exposure_max"]
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=counters,
            completed_sources=idents[:4],
        )
    assert "total_exposure_seconds" in str(exc_info.value)
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_rejects_min_fabrication_when_all_new_unknown(tmp_path):
    """When every new frame is unknown, a fabricated lower min is refused."""
    _commit_gen1(tmp_path, n_sources=6, frame_count=2)  # known: min 1.0
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    counters = _counters(4)
    counters["exposure_unknown_count"] = 2
    counters["total_exposure_seconds"] = cont.counters["total_exposure_seconds"]
    counters["exposure_min"] = cont.counters["exposure_min"] - 0.5  # 0.5
    counters["exposure_max"] = cont.counters["exposure_max"]
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=counters,
            completed_sources=idents[:4],
        )
    assert "exposure_min" in str(exc_info.value)
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_rejects_max_fabrication_when_all_new_unknown(tmp_path):
    """When every new frame is unknown, a fabricated higher max is refused."""
    _commit_gen1(tmp_path, n_sources=6, frame_count=2)  # known: max 1.1
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    counters = _counters(4)
    counters["exposure_unknown_count"] = 2
    counters["total_exposure_seconds"] = cont.counters["total_exposure_seconds"]
    counters["exposure_min"] = cont.counters["exposure_min"]
    counters["exposure_max"] = cont.counters["exposure_max"] + 0.5  # 1.6
    with pytest.raises(DrizzleCheckpointError) as exc_info:
        cont.writer.commit(
            cont.accumulators, session_binding=binding, counters=counters,
            completed_sources=idents[:4],
        )
    assert "exposure_max" in str(exc_info.value)
    _assert_gen1_still_authoritative(tmp_path)


def test_continuation_valid_all_unknown_transition(tmp_path):
    """A fully-unknown loaded run may continue with fully-unknown new frames
    (known_added == 0) only if total/min/max stay exactly unchanged."""
    _commit_gen1(
        tmp_path, n_sources=6, frame_count=2,
        counters_overrides={
            "exposure_unknown_count": 2,
            "total_exposure_seconds": 0.0,
            "exposure_min": None,
            "exposure_max": None,
        },
    )
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    counters = _counters(4)
    counters["exposure_unknown_count"] = 4
    counters["total_exposure_seconds"] = 0.0
    counters["exposure_min"] = None
    counters["exposure_max"] = None
    assert cont.writer.commit(
        cont.accumulators, session_binding=binding, counters=counters,
        completed_sources=idents[:4],
    ) == 2
    manifest = _load_manifest(tmp_path)
    assert manifest["generation"] == 2
    assert manifest["exposure_unknown_count"] == 4
    assert manifest["total_exposure_seconds"] == 0.0
    assert manifest["exposure_min"] is None
    assert manifest["exposure_max"] is None


def test_continuation_valid_mixed_unknown_known_transition(tmp_path):
    """A mixed transition (one known + one unknown new frame) is legal: total
    strictly increases and min/max are known afterwards."""
    _commit_gen1(
        tmp_path, n_sources=6, frame_count=2,
        counters_overrides={
            "exposure_unknown_count": 1,
            "total_exposure_seconds": 1.0,
            "exposure_min": 1.0,
            "exposure_max": 1.0,
        },
    )
    cont, idents, binding = _read_and_rearm(tmp_path)
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    counters = {
        "frame_count": 4,
        "stacked_batches_count": 4,
        "total_exposure_seconds": 2.3,
        "exposure_unknown_count": 2,
        "exposure_min": 1.0,
        "exposure_max": 1.3,
    }
    assert cont.writer.commit(
        cont.accumulators, session_binding=binding, counters=counters,
        completed_sources=idents[:4],
    ) == 2
    manifest = _load_manifest(tmp_path)
    assert manifest["generation"] == 2
    assert manifest["exposure_unknown_count"] == 2
    assert manifest["total_exposure_seconds"] == 2.3
    assert manifest["exposure_min"] == 1.0
    assert manifest["exposure_max"] == 1.3


# ---------------------------------------------------------------------------
# I. exact directory provenance / symlink-root rebind protection (finding 3)
# ---------------------------------------------------------------------------


def test_continuation_symlink_root_rebind_protection(tmp_path):
    """A validated result binds to the canonical real directory; retargeting a
    symlink root afterwards cannot rebind re-arm to another checkpoint."""
    real_a = Path(tmp_path) / "real_a"
    real_b = Path(tmp_path) / "real_b"
    real_a.mkdir()
    real_b.mkdir()

    _commit_gen1(real_a, n_sources=6, frame_count=2)
    _commit_gen1(real_b, n_sources=6, frame_count=2)

    link = Path(tmp_path) / "link"
    try:
        link.symlink_to(real_a, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")

    # Validate through the symlink root; provenance binds to the real path.
    res = read_drizzle_checkpoint(str(link))
    assert res.source_output_dir == os.path.realpath(str(link))
    assert res.source_output_dir == os.path.realpath(str(real_a))

    # Retarget the symlink to the OTHER checkpoint.
    link.unlink()
    link.symlink_to(real_b, target_is_directory=True)

    # Re-arm from the (now retargeted) result: it must stay on real_a, never
    # bind real_b's checkpoint.
    cont = DrizzleCheckpointWriter.from_validated_result(res)
    assert cont.writer.output_dir == os.path.realpath(str(real_a))
    assert cont.writer.output_dir != os.path.realpath(str(real_b))
    assert cont.generation == 1

    idents = list(cont.session["plan"]["sources"])
    binding = {
        "input_roots": cont.session["input_roots"],
        "reference": cont.session["reference"],
        "plan": cont.session["plan"],
    }
    frames = build_frames()
    for f in frames[2:4]:
        _add_frame_to_all(cont.accumulators, f)
    assert cont.writer.commit(
        cont.accumulators, session_binding=binding, counters=_counters(4),
        completed_sources=idents[:4],
    ) == 2

    # real_a advanced to generation 2; real_b stayed generation 1.
    assert _load_manifest(real_a)["generation"] == 2
    assert _load_manifest(real_b)["generation"] == 1


def test_continuation_factory_refuses_symlink_swap_of_provenance(tmp_path):
    """If the validated provenance is later swapped for a symlink to another
    checkpoint, the factory refuses instead of binding the other checkpoint."""
    real_a = Path(tmp_path) / "real_a"
    real_b = Path(tmp_path) / "real_b"
    real_a.mkdir()
    real_b.mkdir()
    _commit_gen1(real_a, n_sources=6, frame_count=2)
    _commit_gen1(real_b, n_sources=6, frame_count=2)

    link = Path(tmp_path) / "link"
    try:
        link.symlink_to(real_a, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")

    res = read_drizzle_checkpoint(str(real_a))
    # Tamper the frozen provenance to point at the symlink (bypassing the
    # frozen __post_init__ realpath normalization).
    object.__setattr__(res, "source_output_dir", str(link))

    # Retarget the symlink to the OTHER checkpoint.
    link.unlink()
    link.symlink_to(real_b, target_is_directory=True)

    with pytest.raises(DrizzleCheckpointError) as exc_info:
        DrizzleCheckpointWriter.from_validated_result(res)
    assert "symlink" in str(exc_info.value)
    # Neither checkpoint was mutated by the refused re-arm.
    assert _load_manifest(real_a)["generation"] == 1
    assert _load_manifest(real_b)["generation"] == 1
