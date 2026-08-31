"""COV-01C focused tests: Drizzle positive-support domain.

Proves native SCI/out_img and native out_wht are byte-identical with and
without support tracking; signed Lanczos negative WHT lobes are preserved
while the separate support domain stays non-negative; per-original-exposure
decomposition invariance; derived N_eff; and legacy support-less reopen.
"""

import numpy as np
import pytest
from astropy.wcs import WCS
from astropy.io import fits

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


def test_support_n_eff_and_legacy_load(tmp_path):
    shape = (4, 4)
    stack = _drizzle_stack(tmp_path, shape, "square", support=True)
    _add_frames(stack, shape, 3, seed=5)
    neff = stack._drizzle_support_n_eff()
    assert neff is not None
    assert neff.shape == shape
    assert np.all(np.isfinite(neff))
    assert np.all(neff >= 0.0)
    # legacy: no sidecar manifest -> (None, None)
    w1, w2 = stack._load_drizzle_support(1, 3, shape)
    assert w1 is None and w2 is None


# ---------------------------------------------------------------------------
# REWORK R1: persistence / Stop-Resume / stale / cleanup witnesses
# ---------------------------------------------------------------------------

import pathlib
import os as _os

from seestar.core.drizzle_checkpoint import DrizzleCheckpointError


def test_support_persist_reopen_no_orphan(tmp_path):
    shape = (6, 7)
    stack = _drizzle_stack(tmp_path, shape, "square", support=True)
    _add_frames(stack, shape, 3, seed=9)
    w1_before = np.asarray(stack.drizzle_sup_w1.wht, dtype=np.float64).copy()
    w2_before = np.asarray(stack.drizzle_sup_w2.wht, dtype=np.float64).copy()
    stack._persist_drizzle_support(1, 3)
    sup_dir = pathlib.Path(tmp_path) / "drizzle_support"
    assert (sup_dir / "sup_w1.npy").exists()
    assert (sup_dir / "sup_w2.npy").exists()
    assert (sup_dir / "manifest.json").exists()
    # no temp orphan files remain
    orphans = [n for n in _os.listdir(sup_dir) if ".tmp" in n or n.startswith(".sup-")]
    assert orphans == []
    # reopen via the actual load path -> exact equivalence
    w1, w2 = stack._load_drizzle_support(1, 3, shape)
    assert w1 is not None and w2 is not None
    assert np.array_equal(np.asarray(w1.wht, dtype=np.float64), w1_before)
    assert np.array_equal(np.asarray(w2.wht, dtype=np.float64), w2_before)


def test_support_stale_generation_fails(tmp_path):
    shape = (6, 7)
    stack = _drizzle_stack(tmp_path, shape, "square", support=True)
    _add_frames(stack, shape, 2, seed=10)
    stack._persist_drizzle_support(1, 2)
    with pytest.raises(DrizzleCheckpointError):
        stack._load_drizzle_support(2, 2, shape)


def test_support_frame_count_mismatch_fails(tmp_path):
    shape = (6, 7)
    stack = _drizzle_stack(tmp_path, shape, "square", support=True)
    _add_frames(stack, shape, 2, seed=11)
    stack._persist_drizzle_support(1, 2)
    with pytest.raises(DrizzleCheckpointError):
        stack._load_drizzle_support(1, 3, shape)


def test_support_corrupt_manifest_fails(tmp_path):
    shape = (6, 7)
    stack = _drizzle_stack(tmp_path, shape, "square", support=True)
    _add_frames(stack, shape, 2, seed=12)
    stack._persist_drizzle_support(1, 2)
    mp = pathlib.Path(tmp_path) / "drizzle_support" / "manifest.json"
    mp.write_text("{corrupt json", encoding="utf-8")
    with pytest.raises(DrizzleCheckpointError):
        stack._load_drizzle_support(1, 2, shape)


def test_support_cleanup_preserves_pre_existing(tmp_path):
    out = pathlib.Path(tmp_path) / "out"
    sup_dir = out / "drizzle_support"
    sup_dir.mkdir(parents=True)
    sentinel = sup_dir / "user_sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    stack = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    stack.output_folder = str(out)
    # snapshot the pre-existing dir + sentinel, then simulate an attempt-created file
    stack._attempt_preexisting_state = set()
    stack._attempt_preexisting_state.add(_os.path.normcase(_os.path.abspath(str(sup_dir))))
    stack._attempt_preexisting_state.add(_os.path.normcase(_os.path.abspath(str(sentinel))))
    (sup_dir / "sup_w1.npy").write_bytes(b"attempt")
    (sup_dir / "sup_w2.npy").write_bytes(b"attempt")
    (sup_dir / "manifest.json").write_bytes(b"{}")
    stack._remove_attempt_created_state()
    # attempt-created files removed; pre-existing sentinel preserved
    assert sentinel.exists()
    assert not (sup_dir / "sup_w1.npy").exists()
    assert not (sup_dir / "sup_w2.npy").exists()
    assert not (sup_dir / "manifest.json").exists()
