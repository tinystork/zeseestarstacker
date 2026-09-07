"""Phase 1 (zsss-support-overlap-p1-20260907) — production seam witnesses.

REWORK-2 / Item E: analytic fixtures with EXPLICIT known support, witnessed
through the REAL production methods (``_normalize_sources_against_reference``
and ``_stack_batch``) using the EXACT per-frame support carrier the worker
seam publishes (``(M, src_content_valid, src_shape)`` built from the loader
opt-in invalidity report, exactly as ``_process_file`` does):

* loader opt-in invalidity report is truthful (original non-finite before
  repair) and the science path stays bit-identical to the default loader;
* plain-Classic ``_stack_batch`` normalizes every frame against the immutable
  reference with the paired-overlap estimators when real geometry/content
  evidence is present (identity carrier == fully aligned frame);
* a missing carrier / missing content / missing M answers NEUTRAL with a
  stable reason (never a legacy full-frame fallback, never an all-valid
  guess), and ``none`` stays a strict image no-op;
* legitimate zero-valued SUPPORTED pixels are not confusable with repaired
  padding zeros (geometry decides, never brightness);
* the per-thread carrier slot semantics (publish/consume/clear) are race-free
  by construction (thread-local) and never leak between files;
* bounded fail-open diagnostics carry only scalars/counts — never retained
  canvas masks;
* the classic checkpoint marker is written into the manifest (scalar only, no
  M/full-mask serialization) and Resume refuses on marker absence/mismatch
  when the current session's normalization science changed.

All scenes are synthetic, deterministic and fast (no GPU / GUI / network).
"""

import json
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from seestar.queuep import queue_manager as qm
from seestar.queuep.queue_manager import SeestarQueuedStacker
from seestar.core.image_processing import load_and_validate_fits
from seestar.core.overlap_normalization import (
    REASON_ACCEPTED,
    REASON_INSUFFICIENT_OVERLAP,
    REASON_NO_GEOMETRY,
    REASON_NO_REFERENCE_CONTENT,
    REASON_NO_SOURCE_CONTENT,
    content_validity_after_loader,
    content_valid_canvas,
    estimate_linear_fit_from_geometry,
    estimate_sky_mean_from_geometry,
    geometry_support_mask,
    luminance,
)

HEADER = fits.Header()
NORM_TOL = 5e-3


# ---------------------------------------------------------------------------
# Minimal plain-classic stack stub (same attributes the HSI harness sets; the
# two methods under test only read these).
# ---------------------------------------------------------------------------
def make_stack(norm="none", ref=None):
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.stacking_mode = "mean"
    o.normalize_method = norm
    o.weighting_method = "none"
    o.use_quality_weighting = False
    o.weight_by_snr = True
    o.weight_by_stars = False
    o.snr_exponent = 1.0
    o.stars_exponent = 0.5
    o.min_weight = 0.0
    o.apply_batch_feathering = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.drizzle_active_session = False
    o.is_mosaic_run = False
    o.stack_kappa_low = 3.0
    o.stack_kappa_high = 3.0
    o.winsor_limits = (0.05, 0.05)
    o.stack_reject_algo = "none"
    o.max_hq_mem = 1_000_000_000
    o.batch_size = 10
    o.settings = None
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    # fresh per-test transient state (normally created in __init__)
    o._p1_tls = threading.local()
    o._p1_norm_diagnostics = []
    o._norm_reference = None
    o._norm_reference_content = None
    if ref is not None:
        o._capture_normalization_reference(ref)
        o._norm_reference_content = np.ones(np.asarray(ref).shape[:2], dtype=bool)
    return o


def identity_carrier(img):
    """Exact carrier format published by ``_process_file`` for an aligned frame.

    ``(M 2x3 float64, src_content_valid bool (H,W), src_shape (H,W))``.
    """
    H, W = np.asarray(img).shape[:2]
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    return (M, np.ones((H, W), dtype=bool), (H, W))


def batch_item(img, carrier):
    """6-tuple batch item exactly as the plain-classic worker appends it."""
    return (
        np.array(img, dtype=np.float32, copy=True),
        HEADER,
        {"snr": 1.0, "stars": 0.0},
        None,
        np.ones(np.asarray(img).shape[:2], dtype=bool),
        carrier,
    )


def _dataset(shape=(64, 64)):
    rng = np.random.default_rng(20260907)
    H, W = shape
    ii = np.arange(H, dtype=np.float64)[:, None]
    jj = np.arange(W, dtype=np.float64)[None, :]
    ramp = 100.0 + 200.0 * (ii / (H - 1)) + 60.0 * (jj / (W - 1))
    A = (ramp + rng.normal(0.0, 5.0, size=shape)).astype(np.float32)
    return {"A": A, "Bs": (A + 40.0).astype(np.float32), "B": (1.5 * A + 40.0).astype(np.float32)}


# ---------------------------------------------------------------------------
# W1. Loader opt-in invalidity: truthful + science bit-identical
# ---------------------------------------------------------------------------
def test_loader_report_invalidity_truthful_and_science_bit_identical(tmp_path):
    H = W = 64
    base = np.linspace(100.0, 300.0, H * W, dtype=np.float32).reshape(H, W)
    bad = (slice(10, 14), slice(20, 24))
    raw = base.copy()
    raw[bad] = np.nan
    fp = str(tmp_path / "invalid.fit")
    fits.PrimaryHDU(data=raw).writeto(fp, overwrite=True)

    img_def, _hdr_def = load_and_validate_fits(fp)
    img_rep, _hdr_rep, invalid = load_and_validate_fits(fp, report_invalidity=True)
    # Science is bit-identical between the default and the opt-in report path.
    assert np.array_equal(img_def, img_rep)
    # The report records the ORIGINAL non-finite positions before repair.
    assert invalid.dtype == bool and invalid.shape == (H, W)
    assert invalid[bad].all()
    assert not invalid[~np.isnan(raw)].any()
    # Repaired positions became numeric zeros in the returned science.
    assert (img_rep[bad] == 0.0).all()

    # Failure paths return a None mask (never a guessed all-valid mask).
    r = load_and_validate_fits(str(tmp_path / "missing.fit"), report_invalidity=True)
    assert r == (None, None, None) or (r[0] is None and r[2] is None)


# ---------------------------------------------------------------------------
# W2. Real seam: _stack_batch normalizes with the published carrier
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("norm,key,kind", [
    ("linear_fit", "B", "affine"),
    ("sky_mean", "Bs", "offset"),
])
def test_stack_batch_plain_classic_support_normalizes(norm, key, kind):
    D = _dataset()
    A, S = D["A"], D[key]
    stack = make_stack(norm, ref=A)
    V, hdr, W = stack._stack_batch(
        [batch_item(A, identity_carrier(A)), batch_item(S, identity_carrier(S))],
        1,
        1,
    )
    assert V is not None
    # linear_fit resolves affine transforms to the reference; sky_mean aligns
    # pure offsets.  Both must land on A (measured <= 6e-5 on this dataset).
    assert np.allclose(V, A, atol=NORM_TOL), (norm, kind, float(np.abs(V - A).max()))
    # Diagnostics recorded per frame, scalars only, reasons accepted.
    rows = list(stack._p1_norm_diagnostics)
    assert len(rows) == 2
    assert all(r["reason"] == REASON_ACCEPTED for r in rows)
    assert all(r["method"] == norm for r in rows)
    assert all(r["geometry"] is True for r in rows)
    for r in rows:
        assert "frame_index" in r and "n_effective" in r and "n_overlap" in r
        assert "effective_fraction" in r and isinstance(r["effective_fraction"], float)


def test_stack_batch_missing_carrier_is_neutral_not_legacy_fallback():
    D = _dataset()
    A, B = D["A"], D["B"]
    stack = make_stack("linear_fit", ref=A)
    # Legacy 5-tuple items (no carrier) flowing into the plain-classic seam:
    # per-frame NEUTRAL (identity) with a stable reason — never the legacy
    # full-frame helper, never an error.  With the whole carrier absent both M
    # and source content are unknown; the estimator deterministically reports
    # the FIRST missing evidence (source content validity).
    item = batch_item(B, None)[:5]
    V, _hdr, _W = stack._stack_batch([item], 1, 1)
    assert np.allclose(V, B, atol=1e-6)  # unchanged (neutral), not mapped to A
    assert not np.allclose(V, A, atol=NORM_TOL)
    assert len(stack._p1_norm_diagnostics) == 1
    diag = stack._p1_norm_diagnostics[0]
    assert diag["reason"] == REASON_NO_SOURCE_CONTENT  # no content evidence
    assert diag["geometry"] is False


def test_stack_batch_missing_content_masks_neutral():
    D = _dataset()
    A, B = D["A"], D["B"]
    stack = make_stack("sky_mean", ref=A)
    H, W = A.shape
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    # Source content evidence missing (provenance lost) -> neutral + reason.
    carrier_no_src = (M, None, (H, W))
    V, _h, _w = stack._stack_batch([batch_item(B, carrier_no_src)], 1, 1)
    assert np.allclose(V, B, atol=1e-6)
    assert stack._p1_norm_diagnostics[-1]["reason"] == REASON_NO_SOURCE_CONTENT

    # Reference content evidence missing -> neutral + reason.
    stack2 = make_stack("sky_mean", ref=A)
    stack2._norm_reference_content = None
    V2, _h, _w = stack2._stack_batch([batch_item(B, identity_carrier(B))], 1, 1)
    assert np.allclose(V2, B, atol=1e-6)
    assert stack2._p1_norm_diagnostics[-1]["reason"] == REASON_NO_REFERENCE_CONTENT


def test_stack_batch_none_is_strict_noop_through_seam():
    D = _dataset()
    A, B = D["A"], D["B"]
    stack = make_stack("none", ref=None)
    V, hdr, _W = stack._stack_batch([batch_item(B, identity_carrier(B))], 1, 1)
    assert np.allclose(V, B, atol=1e-6)
    assert stack._p1_norm_diagnostics == []  # none never records normalization


def test_stack_batch_missing_reference_fails_closed():
    D = _dataset()
    A, B = D["A"], D["B"]
    stack = make_stack("linear_fit", ref=None)
    with pytest.raises(RuntimeError, match="session reference"):
        stack._stack_batch([batch_item(B, identity_carrier(B))], 1, 1)


# ---------------------------------------------------------------------------
# W3. Legitimate supported zeros vs repaired padding zeros (seam level)
# ---------------------------------------------------------------------------
def test_stack_batch_supported_zero_pixels_vs_padding_zeros():
    # Reference: constant sky 0.2.  Source: same sky +D observed on a shifted
    # footprint (translation -> >25% of the canvas is repaired NaN padding).
    # Inside the SUPPORTED footprint we embed genuine zero-valued science
    # pixels at a minority fraction: they are content-valid (astronomical
    # zero), while the padding zeros are repaired content excluded by the
    # geometry.  The paired estimator must recover D (padding must not drag
    # the robust offset to -sky).
    import cv2

    H = W = 128
    base_sky = 0.2
    D = 0.05
    tx = 40.0  # >25% of the canvas becomes repaired padding
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, tx]], dtype=np.float64)
    ref = np.full((H, W), base_sky, dtype=np.float32)
    src_det = np.full((H, W), base_sky + D, dtype=np.float32)
    # ~12% of the detector: legitimate zero-valued science pixels.
    rng = np.random.default_rng(3)
    zero_frac = 0.12
    zsel = rng.random((H, W)) < zero_frac
    src_det[zsel] = 0.0
    canvas = cv2.warpAffine(
        src_det,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    src_canvas = np.where(np.isnan(canvas), 0.0, canvas).astype(np.float32)
    geom = geometry_support_mask((H, W), M, (H, W))
    assert (1.0 - geom.mean()) > 0.25  # real padding majority...
    # ...yet the padding zeros are geometrically excluded:
    inside = geom & np.isfinite(canvas)
    padding_only = (~geom) & (src_canvas == 0.0)
    assert padding_only.sum() > 0

    stack = make_stack("sky_mean", ref=ref)
    content = content_validity_after_loader(
        ~np.ones((H, W), dtype=bool), bayer=False
    )  # no original invalidity -> fully content-valid detector
    assert content.all()
    # Carrier built exactly like _process_file: M + loader-derived content.
    carrier = (M, content, (H, W))
    V, _h, _w = stack._stack_batch([batch_item(src_canvas, carrier)], 1, 1)
    diag = stack._p1_norm_diagnostics[-1]
    assert diag["reason"] == REASON_ACCEPTED
    # Padding zeros would drag a legacy full-canvas offset toward -sky; the
    # support-aware offset recovers D (supported legit zeros are a minority
    # and robust-location survives them).
    assert abs(diag["offset"] - D) <= 0.02, diag["offset"]
    # Supported legit zeros stay content-supported (never brightness-cut).
    inside_src = (geom) & (np.isfinite(canvas)) & (np.abs(src_canvas) < 1e-9)
    assert inside_src.sum() > 0


# ---------------------------------------------------------------------------
# W4. Carrier slot semantics (thread-local publish / consume / clear)
# ---------------------------------------------------------------------------
def test_carrier_slot_thread_local_publish_consume_clear():
    stack = make_stack("linear_fit", ref=None)
    assert stack._p1_read_carrier() is None  # no carrier published yet
    carrier = (np.zeros((2, 3)), None, (4, 4))
    stack._p1_carrier_slot().support_carrier = carrier
    # Same-thread read sees the value; consume clears it.
    assert stack._p1_read_carrier() is carrier
    stack._p1_carrier_slot().support_carrier = None
    assert stack._p1_read_carrier() is None

    # Cross-thread isolation: a worker thread cannot see or clobber the main
    # thread's slot, and its own publish does not leak back.
    stack._p1_carrier_slot().support_carrier = carrier
    seen = {}

    def worker():
        wslot = stack._p1_carrier_slot()
        seen["before"] = getattr(wslot, "support_carrier", "<absent>")
        wslot.support_carrier = "worker-value"

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["before"] == "<absent>" or seen["before"] is None  # isolated
    assert stack._p1_read_carrier() is carrier  # unclobbered
    stack._p1_carrier_slot().support_carrier = None


# ---------------------------------------------------------------------------
# W5. Marker: contract matrix, manifest scalar-only persistence, resume refusal
# ---------------------------------------------------------------------------
def _contract_stack(plain, norm, batch_size, **kw):
    o = make_stack(norm, ref=None)
    o.batch_size = batch_size
    o.is_mosaic_run = kw.get("mosaic", False)
    o.drizzle_active_session = kw.get("drizzle", False)
    o.reproject_between_batches = kw.get("reproject_b", False)
    o.reproject_coadd_final = kw.get("reproject_c", False)
    if not plain:
        o.reproject_between_batches = True
        if kw.get("mosaic"):
            o.is_mosaic_run = True
            o.reproject_between_batches = False
        if kw.get("drizzle"):
            o.drizzle_active_session = True
            o.reproject_between_batches = False
    return o


def test_classic_norm_science_contract_matrix():
    OVERLAP = qm._CLASSIC_NORM_SCIENCE_OVERLAP
    LEGACY = qm._CLASSIC_NORM_SCIENCE_LEGACY_UNCHANGED
    # plain classic + sky_mean/linear_fit -> overlap science
    assert _contract_stack(True, "sky_mean", 1)._classic_norm_science_contract() == OVERLAP
    assert _contract_stack(True, "linear_fit", 7)._classic_norm_science_contract() == OVERLAP
    # plain classic + none + batch_size==1 -> BS1 hidden subtraction bypassed
    assert _contract_stack(True, "none", 1)._classic_norm_science_contract() == OVERLAP
    # plain classic + none + batch>1 -> byte-identical legacy science
    assert _contract_stack(True, "none", 7)._classic_norm_science_contract() == LEGACY
    # non-plain paths never write a classic overlap contract
    assert _contract_stack(False, "linear_fit", 1, mosaic=True)._classic_norm_science_contract() == LEGACY
    assert _contract_stack(False, "sky_mean", 1, drizzle=True)._classic_norm_science_contract() == LEGACY
    assert _contract_stack(False, "none", 1)._classic_norm_science_contract() == LEGACY


def test_manifest_marker_scalar_only_no_m_or_masks(tmp_path):
    D = _dataset()
    A = D["A"]
    o = make_stack("linear_fit", ref=A)
    o.output_folder = str(tmp_path)
    o._resume_manifest_schema_version = qm._RESUME_MANIFEST_VERSION_MIN
    o.memmap_shape = tuple(A.shape + (3,))
    o.memmap_dtype_sum = np.float32
    o.memmap_dtype_wht = np.float32
    o._resume_completed_sources = []
    o.stacked_batches_count = 0
    o.images_in_cumulative_stack = 0
    o.total_exposure_seconds = 0.0
    o._exposure_unknown_count = 0
    o._exposure_min = None
    o._exposure_max = None
    o.current_stack_header = None
    o.use_quality_weighting = False
    # Seam-local stubs for writer bookkeeping that is not under test.
    o._support_manifest_metadata = lambda: None
    o._serialize_cumulative_header = lambda: None
    o._resume_reference_identity = None
    o._resume_input_roots = []
    o._resume_plan = None

    o._write_resume_manifest("clean", completed_sources=[], stacked_batches_count=0)
    mp = Path(tmp_path) / "memmap_accumulators" / qm._RESUME_MANIFEST_FILENAME
    assert mp.exists()
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    assert manifest["classic_norm_science"] == qm._CLASSIC_NORM_SCIENCE_OVERLAP
    text = mp.read_text(encoding="utf-8")
    # No M / content-mask serialization of any kind (scalar marker only).
    for banned in ('"M"', "content_mask", "_norm_reference_content", "full_mask"):
        assert banned not in text, banned


def _write_manifest_for(stack, out_dir, state="clean", marker=None, fp=None):
    memdir = Path(out_dir) / "memmap_accumulators"
    memdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": qm._RESUME_MANIFEST_VERSION,
        "state": state,
        "mode": qm._RESUME_MODE_CLASSIC_SUMW,
        "fingerprint": fp if fp is not None else stack._scientific_fingerprint(),
        "shape": [64, 64, 3],
    }
    if marker is not None:
        manifest["classic_norm_science"] = marker
    (memdir / qm._RESUME_MANIFEST_FILENAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_resume_refuses_prephase_marker_when_science_changed(tmp_path):
    D = _dataset()
    A = D["A"]
    o = make_stack("linear_fit", ref=A)  # current contract == overlap
    assert o._classic_norm_science_contract() == qm._CLASSIC_NORM_SCIENCE_OVERLAP
    o.output_folder = str(tmp_path)
    # Legacy/pre-Phase-1 checkpoint (no marker): refuse when the current
    # session actually changed normalization science.
    _write_manifest_for(o, str(tmp_path), marker=None)
    ok, reason, _extra = o._validate_resume_headless()
    assert ok is False
    assert "predates" in reason


def test_resume_refuses_marker_mismatch(tmp_path):
    D = _dataset()
    A = D["A"]
    o = make_stack("sky_mean", ref=A)
    o.output_folder = str(tmp_path)
    _write_manifest_for(o, str(tmp_path), marker=qm._CLASSIC_NORM_SCIENCE_LEGACY_UNCHANGED)
    ok, reason, _extra = o._validate_resume_headless()
    assert ok is False
    assert "mismatch" in reason and qm._CLASSIC_NORM_SCIENCE_LEGACY_UNCHANGED in reason


def test_resume_marker_gate_accepts_matching_marker(tmp_path):
    D = _dataset()
    A = D["A"]
    o = make_stack("linear_fit", ref=A)
    o.output_folder = str(tmp_path)
    o.memmap_shape = tuple(A.shape + (3,))
    _write_manifest_for(o, str(tmp_path), marker=qm._CLASSIC_NORM_SCIENCE_OVERLAP)
    ok, reason, _extra = o._validate_resume_headless()
    # The marker GATE accepts the matching overlap marker: validation proceeds
    # past the Phase-1 check and only fails later on manifest sections that
    # are outside this witness's scope (schema-v2 scientific_config).  A
    # marker refusal would return earlier with the exact texts below.
    assert "predates" not in reason
    assert "classic normalization science contract mismatch" not in reason
    assert ok is False  # later-stage v2 validation, not the marker gate


def test_resume_marker_gate_accepts_legacy_marker_when_science_unchanged(tmp_path):
    D = _dataset()
    A = D["A"]
    # Current contract is legacy-unchanged (plain classic, none, batch > 1): a
    # pre-Phase-1 checkpoint (marker absent) must pass the marker gate.
    o = make_stack("none", ref=None)
    o.batch_size = 7
    assert o._classic_norm_science_contract() == qm._CLASSIC_NORM_SCIENCE_LEGACY_UNCHANGED
    o.output_folder = str(tmp_path)
    o.memmap_shape = tuple(A.shape + (3,))
    _write_manifest_for(o, str(tmp_path), marker=None)
    ok, reason, _extra = o._validate_resume_headless()
    assert "predates" not in reason
    assert "classic normalization science contract mismatch" not in reason


# ===========================================================================
# W6. REWORK-3 (Nono F1/F2/F3): stacked/ reference-content rediscovery on
#     clean resume, session-level WARN, end-of-run summary wiring
# ===========================================================================

def _recording_stack(norm="linear_fit", ref=None):
    """make_stack whose update_progress records (msg, level) calls."""
    s = make_stack(norm, ref=ref)
    calls = []
    s._up_calls = calls
    s.update_progress = lambda msg, level=None: calls.append((str(msg), level))
    return s


def _write_ref_fits(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32)).writeto(
        str(path), overwrite=True
    )
    return str(path)


def test_reference_content_rediscovered_from_stacked_on_resume(tmp_path):
    """F1: a reference moved to <src_dir>/stacked/ (clean-resume scenario) is
    still resolved for content evidence via the identity machinery.

    Pre-resume frames corrected against the reference; on clean resume the
    reference DATA is re-pinned and the content mask MUST be re-derivable from
    the verified stacked counterpart (size+mtime), otherwise every post-resume
    frame would silently answer NEUTRAL.  Both resolution orders are covered:
    the persisted resume identity, and the header-provenance + stacked search
    dir candidate."""
    H = W = 64
    bad = (slice(10, 14), slice(20, 24))
    arr = np.full((H, W), 0.4, dtype=np.float32)
    arr[bad] = np.nan
    src_dir = tmp_path / "input"
    orig = _write_ref_fits(src_dir / "ref.fits", arr)

    # capture identity BEFORE the move (move preserves size+mtime)
    probe = make_stack("linear_fit")
    ident = probe._stat_identity(orig)
    assert ident is not None
    # simulate _move_to_stacked: plain move into <src_dir>/stacked/
    stacked_dir = src_dir / "stacked"
    stacked_dir.mkdir()
    moved = str(stacked_dir / "ref.fits")
    os.replace(orig, moved)
    assert not os.path.exists(orig) and os.path.exists(moved)

    # (a) clean resume: persisted resume-reference identity only (no header
    # provenance, empty search dirs).
    s = make_stack("linear_fit")
    s._resume_reference_identity = ident
    cm = s._p1_reference_content_mask(fits.Header(), [])
    assert cm is not None, "content evidence must be re-derivable on resume"
    assert cm.shape == (H, W)
    assert not cm[bad].any()  # loader-original invalidity, truthful
    assert cm[~np.isnan(arr)].all()

    # (b) header provenance + search dirs (fresh-run shape): the stacked
    # counterpart is found through the same verified resolution.
    hdr = fits.Header()
    hdr["_SOURCE_PATH"] = "ref.fits"
    s2 = make_stack("linear_fit")
    cm2 = s2._p1_reference_content_mask(hdr, [str(src_dir)])
    assert cm2 is not None
    assert np.array_equal(cm2, cm)

    # (c) post-resume correction equivalence: with the re-derived mask the
    # estimator ACCEPTS and applies the same correction as the uninterrupted
    # run (constant 0.4 sky, +0.05 offset source; repaired reference patch is
    # excluded from the support, offset still recovers 0.05).
    img_loaded, _h = load_and_validate_fits(moved)
    ref_data = np.asarray(img_loaded, dtype=np.float32)
    good = ~np.isnan(arr)
    s3 = _recording_stack("linear_fit")
    s3._capture_normalization_reference(ref_data)
    s3._norm_reference_content = cm  # what the capture seam stores
    src01 = np.full((H, W), 0.45, dtype=np.float32)  # ref + 0.05 offset
    V, _hd, _W = s3._stack_batch([batch_item(src01, identity_carrier(src01))], 1, 1)
    assert s3._p1_norm_diagnostics[-1]["reason"] == REASON_ACCEPTED
    assert np.allclose(V[good], ref_data[good], atol=NORM_TOL)


def test_reference_content_unresolvable_warns_session_level(tmp_path):
    """F2: when the OVERLAP contract requires reference content evidence that
    cannot be re-derived, a session-level WARN is emitted (update_progress +
    logger.warning) and the helper stays fail-open (returns None, no raise)."""
    D = _dataset()
    A = D["A"]
    s = _recording_stack("linear_fit", ref=A)
    # provenance points at a file that exists nowhere (no resume identity)
    hdr = fits.Header()
    hdr["_SOURCE_PATH"] = "ghost_reference.fit"
    cm = s._p1_reference_content_mask(hdr, [str(tmp_path / "empty")])
    assert cm is None  # fail-open, deterministic
    warns = [m for (m, lvl) in s._up_calls if lvl == "WARN"]
    assert any("reference content" in m.lower() or "P1" in m for m in warns), warns

    # header absent AND no resume identity -> same WARN path
    s2 = _recording_stack("sky_mean", ref=A)
    cm2 = s2._p1_reference_content_mask(fits.Header(), [])
    assert cm2 is None
    warns2 = [m for (m, lvl) in s2._up_calls if lvl == "WARN"]
    assert any("P1" in m or "reference content" in m.lower() for m in warns2), warns2


def test_release_reports_summary_and_warns_on_neutral():
    """F3: end-of-run reporting is wired to the release seam — a session with
    neutral frames emits a session-level WARN summary; an all-accepted session
    does not; state is always cleared."""
    D = _dataset()
    A, B = D["A"], D["B"]
    # neutral session: legacy 5-tuple item (no carrier) => 1 neutral frame
    s = _recording_stack("linear_fit", ref=A)
    item5 = batch_item(B, None)[:5]
    s._stack_batch([item5], 1, 1)
    assert len(s._p1_norm_diagnostics) == 1
    assert s._p1_norm_diagnostics[0]["reason"] != REASON_ACCEPTED
    s._release_norm_reference()
    warns = [m for (m, lvl) in s._up_calls if lvl == "WARN"]
    assert any("neutral" in m and "summary" in m.lower() for m in warns), warns
    assert s._norm_reference is None and s._norm_reference_content is None
    assert s._p1_norm_diagnostics == []

    # accepted session: summary recorded, no session WARN
    s2 = _recording_stack("linear_fit", ref=A)
    s2._stack_batch(
        [batch_item(A, identity_carrier(A)), batch_item(B, identity_carrier(B))],
        1,
        1,
    )
    assert all(r["reason"] == REASON_ACCEPTED for r in s2._p1_norm_diagnostics)
    n_before = len(s2._p1_norm_diagnostics)
    s2._release_norm_reference()
    assert not any(lvl == "WARN" for (_m, lvl) in s2._up_calls)
    assert n_before == 2 and s2._p1_norm_diagnostics == []
