"""Deterministic Drizzle native-state continuation proof (RSM2-D0).

Proves that a real :class:`seestar.core.drizzle_core.DrizzleAccumulator` can be
closed around its persisted native ``_out_img``/``_out_wht`` buffers and safely
continue deposition, producing a result bit-identical to a run that never
stopped.  This is the gate that must PASS before any production Drizzle
checkpoint persistence or Resume enablement.

Covered invariant (for six deterministic frames A..F and every mandatory
kernel ``square`` / ``gaussian`` / ``lanczos2``, at split boundaries 2+4,
3+3 and 5+1):

    continuous A+B+C+D+E+F  ==  A+B+C / snapshot / reconstruct / D+E+F

compared bit-exactly on the native ``out_img`` (weighted-mean science), the
native ``out_wht`` (total, *signed* for Lanczos) and ``finalize("divide")``.

The reconstruction uses the new
:meth:`DrizzleAccumulator.from_native_state` seam, which documents and works
around two verified upstream ``drizzle`` 2.2.0 reconstruction gotchas:

* a pre-populated ``out_wht`` with ``exptime == 0`` raises (the accumulated
  ``total_exptime`` must be restored alongside the buffers);
* a pre-populated ``out_wht`` without a matching ``out_ctx`` raises (the
  context bitmap is bookkeeping only and is therefore disabled on the
  reconstructed engine — no science change).

All frames are small and deterministic (24x24 inputs drizzled onto a 32x32
grid through fractional-shift/rotation pixmaps with finite positive weights
and varying exposure times), exercising the real ``drizzle`` 2.2.0 ``add_image``
C path.
"""

import math

import numpy as np
import pytest

from seestar.core.drizzle_core import (
    WEIGHT_EPSILON,
    DrizzleAccumulator,
    support_integrity_violations,
)
from drizzle.resample import Drizzle

# --------------------------------------------------------------------------
# deterministic frame fixture A..F
# --------------------------------------------------------------------------

OUT_SHAPE = (32, 32)
IN_SHAPE = (24, 24)
_OH, _OW = OUT_SHAPE
_IH, _IW = IN_SHAPE
_YY, _XX = np.indices(IN_SHAPE, dtype=np.float64)

# Per-frame geometric transform: ("trans", dx, dy) or ("rot", deg, cx, cy).
# Fractional shifts / rotation exercise real sub-pixel drizzle; several frames
# are large enough to fall partially outside the 32x32 grid (hard coverage
# edges), which is what produces the signed negative Lanczos WHT lobes.
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
    return np.array([[r[0, 0], r[0, 1], t[0]], [r[1, 0], r[1, 1], t[1]]], dtype=np.float64)


def _frame_tf(spec):
    if spec[0] == "trans":
        return _trans_tf(spec[1], spec[2])
    return _rot_tf(spec[1], spec[2], spec[3])


def build_frames():
    """Build six deterministic 2-D frames ``(data, weight_map, pixmap,
    in_grid_mask, exptime)``.

    ``data`` is a deterministic finite float32 field; ``weight_map`` is a
    finite positive float32 constant per frame; ``pixmap`` is the float64
    ``(Ny, Nx, 2)`` mapping of input pixel centres onto the output grid;
    ``in_grid_mask`` masks pixels whose centre falls outside the output grid.
    """
    frames = []
    for i, spec in enumerate(FRAME_TRANSFORMS):
        tf = _frame_tf(spec)
        px = tf[0, 0] * _XX + tf[0, 1] * _YY + tf[0, 2]
        py = tf[1, 0] * _XX + tf[1, 1] * _YY + tf[1, 2]
        pixmap = np.dstack((px, py))
        in_grid = (
            (pixmap[..., 0] >= 0.0)
            & (pixmap[..., 0] < _OW)
            & (pixmap[..., 1] >= 0.0)
            & (pixmap[..., 1] < _OH)
        )
        data = (
            np.sin(_XX / 2.0 + i) * 5.0 + np.cos(_YY / 1.5) * 3.0 + 20.0
        ).astype(np.float32)
        weight_map = np.full(IN_SHAPE, FRAME_WEIGHTS[i], np.float32)
        frames.append((data, weight_map, pixmap, in_grid, FRAME_EXPTIMES[i]))
    return frames


def _add_frames(acc, frames):
    for data, weight_map, pixmap, in_grid, exptime in frames:
        acc.add(
            data,
            weight_map,
            pixmap,
            exptime=exptime,
            in_units="counts",
            in_grid_mask=in_grid,
        )


def _roundtrip_npy(tmp_path, arr, name):
    """Real float32 ``.npy`` write/read roundtrip (exact bit preservation)."""
    path = tmp_path / f"{name}.npy"
    np.save(path, np.asarray(arr))
    return np.load(path)


def _max_abs_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


# --------------------------------------------------------------------------
# 1. continuous == split / snapshot / reconstruct / continue  (full matrix)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("split", SPLITS)
def test_continuous_equals_split_reconstruct(kernel, split, tmp_path):
    frames = build_frames()

    # Continuous reference: all six frames, one accumulator, one order.
    cont = DrizzleAccumulator(OUT_SHAPE, kernel=kernel, pixfrac=1.0)
    _add_frames(cont, frames)
    cont_img = cont._out_img.copy()
    cont_wht = cont._out_wht.copy()
    cont_sci = cont.finalize("divide")

    # Prefix accumulator closed at the split boundary.
    prefix = DrizzleAccumulator(OUT_SHAPE, kernel=kernel, pixfrac=1.0)
    _add_frames(prefix, frames[:split])
    snap_img = prefix._out_img.copy()
    snap_wht = prefix._out_wht.copy()

    # Simulate persistence / reload: real .npy roundtrip, float32 exact.
    loaded_img = _roundtrip_npy(tmp_path, snap_img, f"{kernel}_{split}_img")
    loaded_wht = _roundtrip_npy(tmp_path, snap_wht, f"{kernel}_{split}_wht")

    total_exptime = sum(FRAME_EXPTIMES[:split])
    restored = DrizzleAccumulator.from_native_state(
        OUT_SHAPE,
        loaded_img,
        loaded_wht,
        kernel=kernel,
        pixfrac=1.0,
        fillval="0.0",
        total_exptime=total_exptime,
    )
    _add_frames(restored, frames[split:])

    # Bit-equality: the reconstruction must be indistinguishable from a run
    # that never stopped (the accumulation order is preserved exactly).
    assert np.array_equal(restored._out_img, cont_img), "native out_img differs"
    assert np.array_equal(restored._out_wht, cont_wht), "native out_wht differs"
    assert np.array_equal(restored.finalize("divide"), cont_sci), "finalize differs"

    # Quantified error (must be exactly zero for the bit-equality claim above).
    assert _max_abs_diff(restored._out_img, cont_img) == 0.0
    assert _max_abs_diff(restored._out_wht, cont_wht) == 0.0

    # Restored total exposure time matches the continuous engine.
    assert restored._total_exptime == sum(FRAME_EXPTIMES)
    assert restored._drizzle.total_exptime == cont._drizzle.total_exptime


# --------------------------------------------------------------------------
# 2. Lanczos2: signed native WHT survives the roundtrip bit-exactly
# --------------------------------------------------------------------------


@pytest.mark.parametrize("split", SPLITS)
def test_lanczos2_signed_wht_preserved_exactly_through_roundtrip(split, tmp_path):
    frames = build_frames()

    prefix = DrizzleAccumulator(OUT_SHAPE, kernel="lanczos2", pixfrac=1.0)
    _add_frames(prefix, frames[:split])
    snap_wht = prefix._out_wht.copy()
    snap_img = prefix._out_img.copy()

    # The deterministic geometry must actually create signed negative WHT.
    neg_mask = snap_wht < 0.0
    assert np.any(neg_mask), "lanczos2 geometry did not create a negative WHT sample"
    n_neg = int(np.count_nonzero(neg_mask))

    # Roundtrip must preserve every sample — including negatives — bit-exactly
    # (never abs/clip/reinterpret the signed WHT).
    loaded_img = _roundtrip_npy(tmp_path, snap_img, f"l2_{split}_img")
    loaded_wht = _roundtrip_npy(tmp_path, snap_wht, f"l2_{split}_wht")

    assert loaded_img.dtype == np.float32
    assert loaded_wht.dtype == np.float32
    assert np.array_equal(loaded_wht, snap_wht)
    assert np.array_equal(loaded_img, snap_img)
    assert int(np.count_nonzero(loaded_wht < 0.0)) == n_neg
    # A specific negative sample is preserved with its exact signed value.
    y, x = np.argwhere(neg_mask)[0]
    assert loaded_wht[y, x] == snap_wht[y, x]
    assert float(loaded_wht[y, x]) < 0.0


# --------------------------------------------------------------------------
# 3. source snapshots are never mutated; restored arrays are owned copies
# --------------------------------------------------------------------------


def test_snapshot_not_mutated_and_restore_owns_copies():
    frames = build_frames()

    prefix = DrizzleAccumulator(OUT_SHAPE, kernel="lanczos2", pixfrac=1.0)
    _add_frames(prefix, frames[:3])
    snap_img = prefix._out_img.copy()
    snap_wht = prefix._out_wht.copy()
    img_before = snap_img.copy()
    wht_before = snap_wht.copy()

    # Keep an alias to the *prefix* engine buffers too (must also stay intact).
    prefix_img_ref = prefix._out_img
    prefix_wht_ref = prefix._out_wht
    prefix_img_before = prefix_img_ref.copy()
    prefix_wht_before = prefix_wht_ref.copy()

    restored = DrizzleAccumulator.from_native_state(
        OUT_SHAPE, snap_img, snap_wht, kernel="lanczos2", pixfrac=1.0,
        fillval="0.0", total_exptime=sum(FRAME_EXPTIMES[:3]),
    )

    # Owned copies: the restored accumulator must NOT alias the caller snapshots.
    assert restored._out_img is not snap_img
    assert restored._out_wht is not snap_wht
    assert restored._out_img.flags.owndata
    assert restored._out_wht.flags.owndata

    # finalize() on the restored accumulator must not mutate its buffers, nor
    # the caller snapshots, nor the original prefix engine buffers.
    _ = restored.finalize("divide")
    _ = restored.finalize("none")

    assert np.array_equal(snap_img, img_before)
    assert np.array_equal(snap_wht, wht_before)
    assert np.array_equal(prefix_img_ref, prefix_img_before)
    assert np.array_equal(prefix_wht_ref, prefix_wht_before)

    # Continuing deposition on the restored accumulator mutates only its own
    # buffers — the caller snapshots stay byte-identical.
    _add_frames(restored, frames[3:])
    assert np.array_equal(snap_img, img_before)
    assert np.array_equal(snap_wht, wht_before)
    assert np.array_equal(prefix_img_ref, prefix_img_before)
    assert np.array_equal(prefix_wht_ref, prefix_wht_before)
    # ... while the restored buffers did accumulate.
    assert not np.array_equal(restored._out_img, img_before)
    assert not np.array_equal(restored._out_wht, wht_before)


# --------------------------------------------------------------------------
# 4. invalid restoration state fails closed before any add
# --------------------------------------------------------------------------


def _valid_snapshot():
    frames = build_frames()
    acc = DrizzleAccumulator(OUT_SHAPE, kernel="square", pixfrac=1.0)
    _add_frames(acc, frames[:2])
    return acc._out_img.copy(), acc._out_wht.copy()


def test_restore_rejects_wrong_dtype():
    img, wht = _valid_snapshot()
    with pytest.raises(TypeError):
        DrizzleAccumulator.from_native_state(
            OUT_SHAPE, img.astype(np.float64), wht, total_exptime=sum(FRAME_EXPTIMES[:2])
        )
    with pytest.raises(TypeError):
        DrizzleAccumulator.from_native_state(
            OUT_SHAPE, img, wht.astype(np.float64), total_exptime=sum(FRAME_EXPTIMES[:2])
        )


def test_restore_rejects_wrong_shape_and_channels():
    img, wht = _valid_snapshot()
    with pytest.raises(ValueError):
        DrizzleAccumulator.from_native_state(
            (24, 24), img, wht, total_exptime=sum(FRAME_EXPTIMES[:2])
        )
    # 3-D (H, W, C) is not a valid single-channel native buffer.
    rgb = np.stack([img, img, img], axis=-1)
    with pytest.raises(ValueError):
        DrizzleAccumulator.from_native_state(
            OUT_SHAPE, rgb, wht, total_exptime=sum(FRAME_EXPTIMES[:2])
        )


def test_restore_rejects_non_array():
    _, wht = _valid_snapshot()
    for bad in (None, "not an array", [1.0, 2.0]):
        with pytest.raises(TypeError):
            DrizzleAccumulator.from_native_state(
                OUT_SHAPE, bad, wht, total_exptime=sum(FRAME_EXPTIMES[:2])
            )


def test_restore_rejects_nonfinite():
    img, wht = _valid_snapshot()
    img_bad = img.copy()
    img_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        DrizzleAccumulator.from_native_state(
            OUT_SHAPE, img_bad, wht, total_exptime=sum(FRAME_EXPTIMES[:2])
        )
    wht_bad = wht.copy()
    wht_bad[1, 1] = np.inf
    with pytest.raises(ValueError):
        DrizzleAccumulator.from_native_state(
            OUT_SHAPE, img, wht_bad, total_exptime=sum(FRAME_EXPTIMES[:2])
        )


def test_restore_rejects_inconsistent_total_exptime():
    img, wht = _valid_snapshot()
    assert np.sum(wht) > 0.0
    with pytest.raises(ValueError):
        DrizzleAccumulator.from_native_state(OUT_SHAPE, img, wht, total_exptime=0.0)
    with pytest.raises(ValueError):
        DrizzleAccumulator.from_native_state(OUT_SHAPE, img, wht, total_exptime=-1.0)


def test_restore_rejects_nonfinite_total_exptime():
    """NaN / ±Inf total_exptime is non-physical and must fail closed before any
    upstream Drizzle is constructed (which would otherwise propagate NaN/Inf
    into the engine's exposure bookkeeping)."""
    img, wht = _valid_snapshot()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            DrizzleAccumulator.from_native_state(
                OUT_SHAPE, img, wht, total_exptime=bad
            )


# --------------------------------------------------------------------------
# 5. restored accumulator retains kernel / pixfrac / fillval / add contract
# --------------------------------------------------------------------------


def test_restore_retains_kernel_pixfrac_fillval_and_add_contract():
    img, wht = _valid_snapshot()
    restored = DrizzleAccumulator.from_native_state(
        OUT_SHAPE,
        img,
        wht,
        kernel="gaussian",
        pixfrac=0.7,
        fillval="3.5",
        total_exptime=sum(FRAME_EXPTIMES[:2]),
    )
    assert restored.kernel == "gaussian"
    assert restored.pixfrac == 0.7
    assert restored.fillval == "3.5"
    assert restored.out_shape_hw == OUT_SHAPE
    # The engine sees the same kernel / fillval and the restored exposure time.
    assert restored._drizzle.kernel == "gaussian"
    assert restored._drizzle.fillval == "3.5"
    assert restored._drizzle.total_exptime == sum(FRAME_EXPTIMES[:2])

    # Later deposits still follow the add contract (order-preserving, exposure
    # scaled).  Adding the remaining frames advances the accumulator exactly as
    # the continuous path would.
    frames = build_frames()
    before_img = restored._out_img.copy()
    _add_frames(restored, frames[2:])
    assert restored._total_exptime == sum(FRAME_EXPTIMES)
    assert not np.array_equal(restored._out_img, before_img)
    # Native WHT is finite everywhere and the science on invalid support is 0.
    assert np.all(np.isfinite(restored._out_wht))
    assert support_integrity_violations(restored.finalize("divide"), restored._out_wht) == []


# --------------------------------------------------------------------------
# 6. upstream finding: naive reconstruction is rejected by drizzle 2.2.0
# --------------------------------------------------------------------------


def test_upstream_naive_reconstruction_is_rejected():
    """Document why :meth:`from_native_state` exists.

    The naive reconstruction in the design doc (``Drizzle(out_img=...,
    out_wht=..., kernel=..., fillval=...)``) fails on the installed engine in
    two independent ways, so a plain wrapper reconstruction around restored
    buffers is not sufficient.
    """
    img, wht = _valid_snapshot()

    # (a) zero total exposure time with a non-empty weight array.
    with pytest.raises(ValueError) as exc_info:
        Drizzle(out_img=img, out_wht=wht, kernel="square", fillval="0.0")
    assert "Exposure time cannot be 0" in str(exc_info.value)

    # (b) non-empty weight array without a matching context bitmap.
    with pytest.raises(ValueError) as exc_info:
        Drizzle(out_img=img, out_wht=wht, kernel="square", fillval="0.0", exptime=1.0)
    assert "Inconsistent values of supplied" in str(exc_info.value)


# --------------------------------------------------------------------------
# 7. final science invariant holds on the reconstructed continuations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", KERNELS)
def test_reconstructed_final_science_support_invariant(kernel):
    frames = build_frames()
    cont = DrizzleAccumulator(OUT_SHAPE, kernel=kernel, pixfrac=1.0)
    _add_frames(cont, frames)

    sci = cont.finalize("divide")
    wht = cont._out_wht
    assert np.all(np.isfinite(sci))
    assert np.all(np.isfinite(wht))
    # Invalid native support (WHT <= epsilon, incl. signed negatives) -> 0 sci.
    assert support_integrity_violations(sci, wht) == []
    # No artificial huge values anywhere.
    assert float(np.max(np.abs(sci))) <= float(np.max(np.abs(cont._out_img))) + 1e-3


# --------------------------------------------------------------------------
# 8. a rejected add fails closed: exposure counters and buffers stay coherent
# --------------------------------------------------------------------------


def test_rejected_add_leaves_exposure_counters_and_buffers_unchanged():
    """A deterministic Python-level validation failure must not advance either
    exposure counter nor the native buffers, and the accumulator must remain
    bit-identical to a fresh valid-only accumulator after a later valid add.

    Upstream ``drizzle`` 2.2.0 increments its internal exposure counter
    (``_texptime`` / ``total_exptime``) *before* the pixmap shape check, so a
    naive wrapper would leave the engine desynchronised.  The wrapper snapshots
    and restores that counter on failure, and only advances its own persistable
    counter after a successful deposition.
    """
    frames = build_frames()
    data, weight_map, pixmap, in_grid, exptime = frames[0]

    acc = DrizzleAccumulator(OUT_SHAPE, kernel="square", pixfrac=1.0)
    img_before = acc._out_img.copy()
    wht_before = acc._out_wht.copy()

    # Deterministic Python-level validation failure: pixmap shape mismatch.
    bad_pixmap = np.zeros((_IH - 1, _IW, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        acc.add(
            data,
            weight_map,
            bad_pixmap,
            exptime=exptime,
            in_units="counts",
            in_grid_mask=in_grid,
        )

    # Both the persistable wrapper counter and the engine counter are unchanged.
    assert acc._total_exptime == 0.0
    assert acc._drizzle.total_exptime == 0.0
    # Native buffers were never mutated by the rejected add.
    assert np.array_equal(acc._out_img, img_before)
    assert np.array_equal(acc._out_wht, wht_before)

    # The accumulator remains usable: a subsequent valid add reproduces, bit for
    # bit, the native buffers and exposure accounting of a fresh valid-only
    # accumulator that never saw the rejected frame.
    fresh = DrizzleAccumulator(OUT_SHAPE, kernel="square", pixfrac=1.0)
    fresh.add(
        data, weight_map, pixmap, exptime=exptime, in_units="counts",
        in_grid_mask=in_grid,
    )
    acc.add(
        data, weight_map, pixmap, exptime=exptime, in_units="counts",
        in_grid_mask=in_grid,
    )

    assert np.array_equal(acc._out_img, fresh._out_img)
    assert np.array_equal(acc._out_wht, fresh._out_wht)
    assert acc._total_exptime == fresh._total_exptime == exptime
    assert acc._drizzle.total_exptime == fresh._drizzle.total_exptime == exptime
