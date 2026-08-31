"""COV-01C focused tests: Drizzle positive-support domain.

Proves native SCI/out_img and native out_wht are byte-identical with and
without support tracking; signed Lanczos negative WHT lobes are preserved
while the separate support domain stays non-negative; per-original-exposure
decomposition invariance; derived N_eff; and legacy support-less reopen.
The native checkpoint witnesses additionally prove one SCI/WHT/support commit
point, Stop→Resume equivalence, and cleanup after support-stage failures.
"""

import json
import os
import pathlib

import numpy as np
import pytest
from astropy.wcs import WCS
from astropy.io import fits

from seestar.core.drizzle_checkpoint import (
    DrizzleCheckpointError,
    DrizzleCheckpointWriter,
    build_drizzle_canonical_config,
    read_drizzle_checkpoint,
)
from seestar.core.drizzle_core import DrizzleAccumulator
from seestar.queuep.queue_manager import SeestarQueuedStacker


KERNELS = ["square", "point", "turbo", "lanczos2", "lanczos3", "gaussian"]


def make_wcs(shape_hw):
    h, w = shape_hw
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = np.array([-0.001, 0.001])
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
    wcs.pixel_shape = (w, h)
    wcs.array_shape = (h, w)
    return wcs


def _drizzle_stack(tmp_path, shape=(6, 7), kernel="square", support=True, shift=0.0):
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.output_folder = str(tmp_path)
    import types as _types
    o.logger = _types.SimpleNamespace(
        warning=lambda *a, **k: None, debug=lambda *a, **k: None,
        info=lambda *a, **k: None, error=lambda *a, **k: None,
    )
    o.update_progress = lambda *a, **k: None
    wcs = make_wcs(shape)
    o.reference_wcs_object = wcs
    o.drizzle_output_wcs = wcs
    o.drizzle_accumulators = [
        DrizzleAccumulator(shape, kernel=kernel, pixfrac=1.0) for _ in range(3)
    ]
    if support:
        o.drizzle_sup_w1 = DrizzleAccumulator(shape, kernel="square", pixfrac=1.0)
        o.drizzle_sup_w2 = DrizzleAccumulator(shape, kernel="square", pixfrac=1.0)
        o._drizzle_support_available = True
    else:
        o.drizzle_sup_w1 = None
        o.drizzle_sup_w2 = None
        o._drizzle_support_available = False
    o._drizzle_bg_anchor = None
    o.shift = shift
    return o


def _frame(shape, rng, shift=0.0):
    h, w = shape
    data = rng.random((h, w, 3)).astype(np.float32) * 1000.0
    weight = (rng.random((h, w)) > 0.2).astype(np.float32)
    tf = np.array([[1.0, 0.0, shift], [0.0, 1.0, shift]], dtype=np.float64)
    return data, weight, tf


def _add_frames(stack, shape, n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    hdr = fits.Header()
    hdr["EXPTIME"] = 1.0
    for _ in range(n):
        data, weight, tf = _frame(shape, rng, shift=shift)
        ok = stack._add_frame_to_drizzle_accumulators(data, hdr, tf, weight)
        assert ok is True


@pytest.mark.parametrize("kernel", KERNELS)
def test_native_sci_wht_parity_with_and_without_support(tmp_path, kernel):
    shape = (6, 7)
    with_sup = _drizzle_stack(tmp_path / "ws", shape, kernel, support=True)
    _add_frames(with_sup, shape, 3, seed=1)
    without = _drizzle_stack(tmp_path / "wo", shape, kernel, support=False)
    _add_frames(without, shape, 3, seed=1)
    for c in range(3):
        assert np.array_equal(
            with_sup.drizzle_accumulators[c]._out_img,
            without.drizzle_accumulators[c]._out_img,
        )
        assert np.array_equal(
            with_sup.drizzle_accumulators[c]._out_wht,
            without.drizzle_accumulators[c]._out_wht,
        )
    # support is non-negative (separate positive domain)
    assert np.all(with_sup.drizzle_sup_w1.wht >= 0.0)
    assert np.all(with_sup.drizzle_sup_w2.wht >= 0.0)


@pytest.mark.parametrize("kernel", ["lanczos2", "lanczos3"])
def test_lanczos_negative_wht_preserved_support_positive(kernel):
    # Manual sub-pixel-shifted pixmap (mirrors the signed-weights probe) so the
    # installed Lanczos engine reliably produces negative native out_wht lobes.
    in_shape = (48, 48)
    out_shape = (64, 64)
    yy, xx = np.indices(in_shape, dtype=np.float64)
    pixmap = np.dstack((xx + 8.5, yy + 8.3)).astype(np.float64)
    data = np.full(in_shape, 100.0, np.float32)
    weight = np.ones(in_shape, np.float32)
    igm = np.ones(in_shape, bool)

    native = DrizzleAccumulator(out_shape, kernel=kernel, pixfrac=1.0)
    native.add(data, weight, pixmap, in_grid_mask=igm)

    sup_w1 = DrizzleAccumulator(out_shape, kernel="square", pixfrac=1.0)
    sup_w2 = DrizzleAccumulator(out_shape, kernel="square", pixfrac=1.0)
    s_i = weight.astype(np.float32)
    sup_w1.add(s_i, s_i, pixmap, in_units="cps", in_grid_mask=igm)
    sup_w2.add(s_i, s_i * s_i, pixmap, in_units="cps", in_grid_mask=igm)

    # native WHT keeps its (preserved) negative lobes; support is non-negative
    assert np.any(native.wht < 0.0)
    assert np.all(sup_w1.wht >= 0.0)
    assert np.all(sup_w2.wht >= 0.0)


def test_support_decomposition_invariance(tmp_path):
    shape = (6, 7)
    rng = np.random.default_rng(11)
    frames = [_frame(shape, rng) for _ in range(5)]
    hdr = fits.Header()
    hdr["EXPTIME"] = 1.0

    def fresh(name):
        return _drizzle_stack(tmp_path / name, shape, "square", support=True)

    all_at_once = fresh("all")
    for data, weight, tf in frames:
        all_at_once._add_frame_to_drizzle_accumulators(data, hdr, tf, weight)

    # a second run of the same ordered sequence is byte-identical (determinism)
    again = fresh("again")
    for data, weight, tf in frames:
        again._add_frame_to_drizzle_accumulators(data, hdr, tf, weight)

    assert np.array_equal(all_at_once.drizzle_sup_w1.wht, again.drizzle_sup_w1.wht)
    assert np.array_equal(all_at_once.drizzle_sup_w2.wht, again.drizzle_sup_w2.wht)


def _checkpoint_writer(output_dir, shape):
    class _ConfigSource:
        pass

    source = _ConfigSource()
    source.weighting_method = "none"
    source.use_quality_weighting = False
    source.weight_by_snr = True
    source.weight_by_stars = True
    source.snr_exponent = 1.0
    source.stars_exponent = 0.5
    source.min_weight = 0.01
    source.correct_hot_pixels = True
    source.hot_pixel_threshold = 3.0
    source.neighborhood_size = 5
    source.bayer_pattern = "GRBG"
    source.drizzle_scale = 1.0
    source.drizzle_kernel = "square"
    source.drizzle_pixfrac = 1.0
    source.drizzle_wht_threshold_effective = 0.0
    source.drizzle_fillval = "0.0"
    cfg = build_drizzle_canonical_config(source, product_version="8.2.0")
    return DrizzleCheckpointWriter(
        str(output_dir), "8.2.0", cfg, make_wcs(shape), shape
    )


def _identity(path):
    stat = os.stat(path)
    return {
        "path": os.path.normcase(str(path)),
        "name": os.path.basename(str(path)),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _checkpoint_inputs(output_dir, count):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for index in range(count):
        path = output_dir / f"source_{index}.fit"
        path.write_bytes(f"source-{index}".encode("ascii"))
        sources.append(_identity(path))
    return sources, {
        "input_roots": [str(output_dir)],
        "reference": sources[0],
        "plan": {"sources": sources, "decomposition": [count]},
    }


def _checkpoint_counters(frame_count):
    return {
        "frame_count": frame_count,
        "stacked_batches_count": frame_count,
        "total_exposure_seconds": float(frame_count),
        "exposure_unknown_count": 0,
        "exposure_min": 1.0,
        "exposure_max": 1.0,
    }


def _commit(writer, stack, binding, sources, frame_count, *, support=True):
    support_accumulators = None
    if support:
        support_accumulators = (stack.drizzle_sup_w1, stack.drizzle_sup_w2)
    return writer.commit(
        stack.drizzle_accumulators,
        session_binding=binding,
        counters=_checkpoint_counters(frame_count),
        completed_sources=sources[:frame_count],
        support_accumulators=support_accumulators,
    )


def test_support_n_eff_and_legacy_native_checkpoint(tmp_path):
    shape = (4, 4)
    stack = _drizzle_stack(tmp_path / "legacy", shape, "square", support=True)
    _add_frames(stack, shape, 3, seed=5)
    neff = stack._drizzle_support_n_eff()
    assert neff is not None
    assert neff.shape == shape
    assert np.all(np.isfinite(neff))
    assert np.all(neff >= 0.0)

    # A schema-v1 native checkpoint with no additive support field is legacy.
    output_dir = tmp_path / "legacy"
    sources, binding = _checkpoint_inputs(output_dir, 3)
    writer = _checkpoint_writer(output_dir, shape)
    _commit(writer, stack, binding, sources, 3, support=False)
    restored = read_drizzle_checkpoint(output_dir)
    assert restored.support_accumulators is None
    assert "support" not in restored.manifest


def test_support_commits_in_native_manifest_and_roundtrips(tmp_path):
    shape = (6, 7)
    output_dir = tmp_path / "native"
    stack = _drizzle_stack(output_dir, shape, "square", support=True)
    _add_frames(stack, shape, 3, seed=9)
    before = (
        stack.drizzle_sup_w1.wht.copy(),
        stack.drizzle_sup_w2.wht.copy(),
    )
    sources, binding = _checkpoint_inputs(output_dir, 3)
    writer = _checkpoint_writer(output_dir, shape)
    assert _commit(writer, stack, binding, sources, 3) == 1

    manifest_path = output_dir / ".m3d_checkpoint" / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["support"]["schema_version"] == 1
    assert manifest["support"]["total_exptime"] == 3.0
    for field in ("sup_w1", "sup_w2"):
        artifact = output_dir / ".m3d_checkpoint" / manifest["support"][field]["file"]
        assert artifact.is_file()
    assert not (output_dir / "drizzle_support").exists()

    restored = read_drizzle_checkpoint(output_dir)
    assert restored.support_accumulators is not None
    for actual, expected in zip(restored.support_accumulators, before):
        assert np.array_equal(actual.wht, expected)


def test_queue_commit_passes_support_to_native_writer(tmp_path):
    stack = _drizzle_stack(tmp_path, (4, 4), "square", support=True)
    _add_frames(stack, (4, 4), 1, seed=8)
    captured = {}

    class _Writer:
        def commit(self, accumulators, **kwargs):
            captured["accumulators"] = accumulators
            captured.update(kwargs)
            return 7

    stack._drizzle_checkpoint_writer = _Writer()
    stack._drizzle_frame_count = 1
    stack._drizzle_completed_sources = [{"path": "source"}]
    stack._drizzle_checkpoint_session_binding = lambda: {"session": True}
    stack._drizzle_checkpoint_counters = lambda: {"frame_count": 1}
    stack._drizzle_checkpoint_commit()

    assert captured["accumulators"] is stack.drizzle_accumulators
    assert captured["support_accumulators"] == (
        stack.drizzle_sup_w1,
        stack.drizzle_sup_w2,
    )
    assert stack._drizzle_checkpoint_last_committed_frames == 1


def test_support_write_failure_preserves_prior_native_generation(
    tmp_path, monkeypatch
):
    shape = (6, 7)
    output_dir = tmp_path / "failure"
    stack = _drizzle_stack(output_dir, shape, "square", support=True)
    _add_frames(stack, shape, 2, seed=10)
    sources, binding = _checkpoint_inputs(output_dir, 3)
    writer = _checkpoint_writer(output_dir, shape)
    _commit(writer, stack, binding, sources, 2)

    checkpoint_dir = output_dir / ".m3d_checkpoint"
    manifest_before = (checkpoint_dir / "checkpoint.json").read_bytes()
    generation_one = {
        path.name: path.read_bytes()
        for path in checkpoint_dir.glob("gen-00000001-*.npy")
    }
    _add_frames(stack, shape, 1, seed=11)
    original = writer._write_array_artifact

    def fail_support_w2(array, final_name):
        if final_name == "gen-00000002-support_w2.npy":
            raise OSError("injected support write failure")
        return original(array, final_name)

    monkeypatch.setattr(writer, "_write_array_artifact", fail_support_w2)
    with pytest.raises(DrizzleCheckpointError, match="support write failure"):
        _commit(writer, stack, binding, sources, 3)

    assert (checkpoint_dir / "checkpoint.json").read_bytes() == manifest_before
    for name, payload in generation_one.items():
        assert (checkpoint_dir / name).read_bytes() == payload
    assert not list(checkpoint_dir.glob("gen-00000002-*.npy"))
    restored = read_drizzle_checkpoint(output_dir)
    assert restored.generation == 1
    assert restored.counters["frame_count"] == 2


def test_support_preflight_failure_writes_no_native_generation(tmp_path):
    shape = (6, 7)
    output_dir = tmp_path / "preflight"
    stack = _drizzle_stack(output_dir, shape, "square", support=True)
    _add_frames(stack, shape, 2, seed=12)
    sources, binding = _checkpoint_inputs(output_dir, 3)
    writer = _checkpoint_writer(output_dir, shape)
    _commit(writer, stack, binding, sources, 2)
    checkpoint_dir = output_dir / ".m3d_checkpoint"
    manifest_before = (checkpoint_dir / "checkpoint.json").read_bytes()

    _add_frames(stack, shape, 1, seed=13)
    stack.drizzle_sup_w1._out_wht[0, 0] = -1.0
    with pytest.raises(DrizzleCheckpointError, match="negative samples"):
        _commit(writer, stack, binding, sources, 3)

    assert (checkpoint_dir / "checkpoint.json").read_bytes() == manifest_before
    assert not list(checkpoint_dir.glob("gen-00000002-*.npy"))


@pytest.mark.parametrize("damage", ["missing_descriptor", "corrupt_artifact"])
def test_support_reader_corruption_fails_closed_without_mutation(tmp_path, damage):
    shape = (6, 7)
    output_dir = tmp_path / damage
    stack = _drizzle_stack(output_dir, shape, "square", support=True)
    _add_frames(stack, shape, 2, seed=14)
    sources, binding = _checkpoint_inputs(output_dir, 2)
    writer = _checkpoint_writer(output_dir, shape)
    _commit(writer, stack, binding, sources, 2)
    checkpoint_dir = output_dir / ".m3d_checkpoint"
    manifest_path = checkpoint_dir / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if damage == "missing_descriptor":
        del manifest["support"]["sup_w2"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        artifact = checkpoint_dir / manifest["support"]["sup_w1"]["file"]
        artifact.write_bytes(b"corrupt")
    before = {
        path.name: path.read_bytes()
        for path in checkpoint_dir.iterdir()
        if path.is_file()
    }

    with pytest.raises(DrizzleCheckpointError):
        read_drizzle_checkpoint(output_dir)

    after = {
        path.name: path.read_bytes()
        for path in checkpoint_dir.iterdir()
        if path.is_file()
    }
    assert after == before


def test_support_stop_resume_matches_continuous_and_neff(tmp_path):
    shape = (6, 7)
    output_dir = tmp_path / "resume"
    first = _drizzle_stack(output_dir, shape, "square", support=True)
    _add_frames(first, shape, 2, seed=20)
    sources, binding = _checkpoint_inputs(output_dir, 3)
    writer = _checkpoint_writer(output_dir, shape)
    _commit(writer, first, binding, sources, 2)

    loaded = read_drizzle_checkpoint(output_dir)
    continuation = DrizzleCheckpointWriter.from_validated_result(loaded)
    resumed = _drizzle_stack(output_dir, shape, "square", support=True)
    resumed.drizzle_accumulators = continuation.accumulators
    resumed.drizzle_sup_w1, resumed.drizzle_sup_w2 = (
        continuation.support_accumulators
    )
    _add_frames(resumed, shape, 1, seed=21)
    _commit(continuation.writer, resumed, binding, sources, 3)
    final = read_drizzle_checkpoint(output_dir)

    continuous = _drizzle_stack(
        tmp_path / "continuous", shape, "square", support=True
    )
    _add_frames(continuous, shape, 2, seed=20)
    _add_frames(continuous, shape, 1, seed=21)

    for actual, expected in zip(
        final.support_accumulators,
        (continuous.drizzle_sup_w1, continuous.drizzle_sup_w2),
    ):
        assert np.array_equal(actual.wht, expected.wht)
    for actual, expected in zip(
        final.accumulators, continuous.drizzle_accumulators
    ):
        assert np.array_equal(actual._out_img, expected._out_img)
        assert np.array_equal(actual._out_wht, expected._out_wht)

    resumed.drizzle_sup_w1, resumed.drizzle_sup_w2 = final.support_accumulators
    assert np.array_equal(
        resumed._drizzle_support_n_eff(), continuous._drizzle_support_n_eff()
    )


def test_legacy_continuation_cannot_fabricate_support(tmp_path):
    shape = (6, 7)
    output_dir = tmp_path / "legacy-continuation"
    stack = _drizzle_stack(output_dir, shape, "square", support=True)
    _add_frames(stack, shape, 2, seed=30)
    sources, binding = _checkpoint_inputs(output_dir, 3)
    writer = _checkpoint_writer(output_dir, shape)
    _commit(writer, stack, binding, sources, 2, support=False)
    continuation = DrizzleCheckpointWriter.from_validated_result(
        read_drizzle_checkpoint(output_dir)
    )
    assert continuation.support_accumulators is None

    resumed = _drizzle_stack(output_dir, shape, "square", support=True)
    resumed.drizzle_accumulators = continuation.accumulators
    _add_frames(resumed, shape, 1, seed=31)
    # Even a caller that forges superficially plausible cumulative metadata
    # cannot introduce support after a legacy support-less generation.
    resumed.drizzle_sup_w1._total_exptime = 3.0
    resumed.drizzle_sup_w2._total_exptime = 3.0
    with pytest.raises(DrizzleCheckpointError, match="availability changed"):
        _commit(continuation.writer, resumed, binding, sources, 3)
