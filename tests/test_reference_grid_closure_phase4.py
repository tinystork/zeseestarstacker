"""GAR-06 Phase 4 acceptance: REFERENCE_GRID_RESUME_ACCEPT.

A checkpoint persists the exact output grid *and* the versioned frozen
input-reference geometry.  A resume restores them exactly (never reconstructs
a new grid under accumulated SCI/WHT).  Legacy north-up / raw-CRPIX
checkpoints are refused clearly, before any mutation, and are never silently
migrated.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

import seestar.queuep.queue_manager as qm
from seestar.core.drizzle_checkpoint import (
    CHECKPOINT_DIRNAME,
    INPUT_REFERENCE_GEOMETRY_CONTRACT,
    MANIFEST_FILENAME,
    RUN_CONFIG_FILENAME,
    DrizzleCheckpointError,
    DrizzleCheckpointWriter,
    build_drizzle_canonical_config,
    reconstruct_input_reference_wcs,
    read_drizzle_checkpoint,
    serialize_input_reference_geometry,
)
from seestar.core.drizzle_core import DrizzleAccumulator, build_output_grid

OUT_SHAPE = (32, 32)
IN_SHAPE = (24, 24)


def _fake_qm(kernel="square"):
    class _Qm:
        pass

    qmobj = _Qm()
    qmobj.weighting_method = "none"
    qmobj.use_quality_weighting = False
    qmobj.weight_by_snr = True
    qmobj.weight_by_stars = True
    qmobj.snr_exponent = 1.0
    qmobj.stars_exponent = 0.5
    qmobj.min_weight = 0.01
    qmobj.correct_hot_pixels = True
    qmobj.hot_pixel_threshold = 3.0
    qmobj.neighborhood_size = 5
    qmobj.bayer_pattern = "GRBG"
    qmobj.drizzle_scale = 1.0
    qmobj.drizzle_kernel = kernel
    qmobj.drizzle_pixfrac = 1.0
    qmobj.drizzle_wht_threshold_effective = 0.0
    qmobj.drizzle_fillval = "0.0"
    return qmobj


def _identity(path):
    st = os.stat(path)
    return {
        "path": os.path.normcase(str(path)),
        "name": os.path.basename(str(path)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _make_wcs(shape_hw, crval=(10.0, 20.0)):
    h, w = shape_hw
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = np.array([-0.001, 0.001])
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
    wcs.array_shape = shape_hw
    return wcs


def _frames(shape_hw, n):
    h, w = shape_hw
    yy, xx = np.indices(shape_hw, dtype=np.float64)
    out = []
    for i in range(n):
        tf = np.array([[1.0, 0.0, 0.3 * i], [0.0, 1.0, -0.2 * i]])
        px = tf[0, 0] * xx + tf[0, 1] * yy + tf[0, 2]
        py = tf[1, 0] * xx + tf[1, 1] * yy + tf[1, 2]
        pixmap = np.dstack((px, py))
        mask = (
            (pixmap[..., 0] >= 0.0)
            & (pixmap[..., 0] < shape_hw[1])
            & (pixmap[..., 1] >= 0.0)
            & (pixmap[..., 1] < shape_hw[0])
        )
        data = (np.sin(xx / 3.0 + i) + 10.0).astype(np.float32)
        weight = np.full(shape_hw, 0.8, np.float32)
        out.append((data, weight, pixmap, mask))
    return out


def _write(tmp_path, n_sources=4, frame_count=2):
    ref_wcs = _make_wcs(IN_SHAPE)
    out_wcs, out_shape_hw = build_output_grid(_make_wcs(OUT_SHAPE), OUT_SHAPE, 1.0)
    cfg = build_drizzle_canonical_config(_fake_qm(), product_version="8.5.0")
    writer = DrizzleCheckpointWriter(
        str(tmp_path), "8.5.0", cfg, out_wcs, out_shape_hw
    )
    accs = [
        DrizzleAccumulator(out_shape_hw, kernel="square", pixfrac=1.0)
        for _ in range(3)
    ]
    frames = _frames(OUT_SHAPE, n_sources)
    for data, weight, pixmap, mask in frames[:frame_count]:
        for acc in accs:
            acc.add(data, weight, pixmap, exptime=1.0, in_units="counts",
                    in_grid_mask=mask)

    ref_path = Path(tmp_path) / "reference.fit"
    ref_path.write_bytes(b"reference-bytes")
    ref_ident = _identity(ref_path)
    src_paths = []
    for i in range(n_sources):
        p = Path(tmp_path) / f"src_{i}.fit"
        p.write_bytes(b"src-%d" % i)
        src_paths.append(p)
    src_idents = [_identity(p) for p in src_paths]

    ref_geometry = serialize_input_reference_geometry(ref_wcs, IN_SHAPE, ref_ident)
    binding = {
        "input_roots": [str(tmp_path)],
        "reference": ref_ident,
        "plan": {"sources": src_idents, "decomposition": [n_sources]},
        "reference_geometry": ref_geometry,
    }
    counters = {
        "frame_count": frame_count,
        "stacked_batches_count": frame_count,
        "total_exposure_seconds": float(frame_count),
        "exposure_unknown_count": 0,
        "exposure_min": 1.0,
        "exposure_max": 1.0,
    }
    writer.commit(
        accs,
        session_binding=binding,
        counters=counters,
        completed_sources=src_idents[:frame_count],
    )
    return {
        "writer": writer,
        "cfg": cfg,
        "ref_wcs": ref_wcs,
        "ref_geometry": ref_geometry,
        "out_wcs": out_wcs,
        "out_shape_hw": out_shape_hw,
    }


def test_input_reference_geometry_persisted_and_restored(tmp_path):
    ctx = _write(tmp_path)
    result = read_drizzle_checkpoint(str(tmp_path))

    assert result.session.get("reference_geometry") is not None
    assert (
        result.reference_geometry["contract"] == INPUT_REFERENCE_GEOMETRY_CONTRACT
    )
    # output grid restored exactly
    assert tuple(result.output_shape_hw) == tuple(ctx["out_shape_hw"])
    assert np.allclose(result.wcs.pixel_scale_matrix, ctx["out_wcs"].pixel_scale_matrix)
    assert np.allclose(result.wcs.wcs.crpix, ctx["out_wcs"].wcs.crpix)
    assert np.allclose(result.wcs.wcs.crval, ctx["out_wcs"].wcs.crval)

    # input-reference WCS restored exactly (geometry, not just re-solving)
    restored_ref = reconstruct_input_reference_wcs(result.reference_geometry)
    assert restored_ref is not None
    assert np.allclose(
        restored_ref.pixel_scale_matrix, ctx["ref_wcs"].pixel_scale_matrix
    )
    assert np.allclose(restored_ref.wcs.crpix, ctx["ref_wcs"].wcs.crpix)
    assert np.allclose(restored_ref.wcs.crval, ctx["ref_wcs"].wcs.crval)
    assert list(restored_ref.wcs.ctype) == list(ctx["ref_wcs"].wcs.ctype)


def test_prefix_restore_identity_new_object(tmp_path):
    ctx = _write(tmp_path)
    first = read_drizzle_checkpoint(str(tmp_path))
    second = read_drizzle_checkpoint(str(tmp_path))  # new object / process read

    W = ctx["writer"]
    for result in (first, second):
        assert tuple(result.output_shape_hw) == tuple(ctx["out_shape_hw"])
        assert result.generation == W.current_generation
        assert result.next_source_index == 2
        assert np.allclose(result.wcs.pixel_scale_matrix, ctx["out_wcs"].pixel_scale_matrix)
    # bit-exact native arrays across re-reads
    for a, b in zip(first.accumulators, second.accumulators):
        assert np.array_equal(a._out_img, b._out_img)
        assert np.array_equal(a._out_wht, b._out_wht)
    # reference geometry is invariant across re-reads
    assert first.reference_geometry == second.reference_geometry


def test_legacy_north_up_checkpoint_is_refused_clearly(tmp_path):
    _write(tmp_path)
    cfg_path = Path(tmp_path) / RUN_CONFIG_FILENAME
    text = cfg_path.read_text(encoding="utf-8")
    assert "m3_output_grid_v2" in text
    legacy = text.replace("m3_output_grid_v2", "m3_output_grid_v1")
    legacy = re.sub(
        r'(output_grid_contract_version"?\s*[=:]\s*)\d+', r"\g<1>1", legacy
    )
    cfg_path.write_text(legacy, encoding="utf-8")

    with pytest.raises(DrizzleCheckpointError) as exc:
        read_drizzle_checkpoint(str(tmp_path))
    assert "legacy Drizzle output-grid checkpoint" in str(exc.value)


def test_missing_reference_geometry_is_rejected(tmp_path):
    _write(tmp_path)
    manifest_path = Path(tmp_path) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["session"]["reference_geometry"]["contract"] = "bogus"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    with pytest.raises(DrizzleCheckpointError):
        read_drizzle_checkpoint(str(tmp_path))


def test_removed_reference_geometry_key_is_rejected(tmp_path):
    """F1: removing the payload (not merely corrupting the token) must refuse."""
    _write(tmp_path)
    manifest_path = Path(tmp_path) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["session"]["reference_geometry"]
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    with pytest.raises(DrizzleCheckpointError) as exc:
        read_drizzle_checkpoint(str(tmp_path))
    assert "input-reference geometry" in str(exc.value)


def test_fresh_commit_without_reference_geometry_fails_closed(tmp_path):
    """F1: production checkpoint creation refuses to publish without it."""
    ref_wcs = _make_wcs(IN_SHAPE)
    out_wcs, out_shape_hw = build_output_grid(_make_wcs(OUT_SHAPE), OUT_SHAPE, 1.0)
    cfg = build_drizzle_canonical_config(_fake_qm(), product_version="8.5.0")
    writer = DrizzleCheckpointWriter(str(tmp_path), "8.5.0", cfg, out_wcs, out_shape_hw)
    accs = [DrizzleAccumulator(out_shape_hw, kernel="square", pixfrac=1.0) for _ in range(3)]

    ref_path = Path(tmp_path) / "reference.fit"
    ref_path.write_bytes(b"reference-bytes")
    src_paths = []
    for i in range(4):
        p = Path(tmp_path) / f"src_{i}.fit"
        p.write_bytes(b"src-%d" % i)
        src_paths.append(p)
    src_idents = [_identity(p) for p in src_paths]
    binding = {
        "input_roots": [str(tmp_path)],
        "reference": _identity(ref_path),
        "plan": {"sources": src_idents, "decomposition": [4]},
    }
    counters = {
        "frame_count": 1, "stacked_batches_count": 1,
        "total_exposure_seconds": 1.0, "exposure_unknown_count": 0,
        "exposure_min": 1.0, "exposure_max": 1.0,
    }
    with pytest.raises(DrizzleCheckpointError) as exc:
        writer.commit(
            accs, session_binding=binding, counters=counters,
            completed_sources=src_idents[:1],
        )
    assert "input-reference geometry is mandatory" in str(exc.value)
    # nothing was published
    assert not (Path(tmp_path) / CHECKPOINT_DIRNAME).exists()


def test_continuation_never_downgrades_reference_geometry(tmp_path):
    """F1: a continuation that omits the payload carries the loaded one forward."""
    ctx = _write(tmp_path)
    result = read_drizzle_checkpoint(str(tmp_path))
    original = result.reference_geometry
    assert original is not None

    from seestar.core.drizzle_checkpoint import DrizzleCheckpointWriter

    cont = DrizzleCheckpointWriter.from_validated_result(result)
    # Continuation binding deliberately omits reference_geometry.
    cont_binding = {
        "input_roots": result.session["input_roots"],
        "reference": result.session["reference"],
        "plan": result.session["plan"],
    }
    from seestar.core.drizzle_checkpoint import DrizzleAccumulator as _Acc

    accs = cont.accumulators
    writer = cont.writer
    # advance one frame using the same frame generator as the initial commit
    data, weight, pixmap, mask = _frames(OUT_SHAPE, 4)[2]
    for acc in accs:
        acc.add(data, weight, pixmap, exptime=1.0, in_units="counts", in_grid_mask=mask)
    counters = {
        "frame_count": 3, "stacked_batches_count": 3,
        "total_exposure_seconds": 3.0, "exposure_unknown_count": 0,
        "exposure_min": 1.0, "exposure_max": 1.0,
    }
    writer.commit(
        accs,
        session_binding=cont_binding,
        counters=counters,
        completed_sources=list(result.session["plan"]["sources"][:3]),
    )
    again = read_drizzle_checkpoint(str(tmp_path))
    assert again.reference_geometry == original


def test_resume_identity_wins_over_conflicting_manual_request():
    class _Dummy:
        pass

    st = _Dummy()
    st._resume_requested = True
    st.reference_origin_hint = None
    # Even with a conflicting new manual request, resume precedence wins.
    assert (
        qm.SeestarQueuedStacker._reference_requested_policy(st, "conflicting.fit")
        == "resume"
    )
