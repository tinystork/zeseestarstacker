"""8.4.0 pre-W80 stage A — durable normalization diagnostics.

Mission ``zsss-840-prew80-20260907``.  Persist EVERY support-aware
plain-Classic ``sky_mean`` / ``linear_fit`` normalization accepted/neutral
event into a per-run append-only ``normalization_diagnostics_<token>.jsonl``
artifact (see :mod:`seestar.core.normalization_diagnostics`), carrying the
REAL original FITS basename (header ``_SRCFILE`` evidence kept lockstep with
the accepted images) — never a synthetic ``batch_frame_i`` identity.

Scope witnesses:

* real source filename in accepted sky_mean, accepted linear_fit and neutral
  fallback events;
* the production ``_stack_batch`` seam attributes surviving frames correctly
  after a filtered bad item (no index-to-path drift);
* valid JSONL, append-only within a run, per-run isolation across stacker
  reuse / resume (distinct session-suffixed artifacts, no overwrite);
* allowlisted bounded scalar records; rejection of arbitrary arrays/matrices;
  finite JSON-safe values; bounded record size;
* RGB bounded a/b channel vectors; documented support fraction / sampled
  count semantics (n_overlap = sampled, n_effective = full common,
  n_geometric = geometry count; unknown support explicitly unknown);
* diagnostics fail-open: writer denied / serialization failure can never
  change normalization output or abort a run (SCI/WHT bit-identical for
  working vs disabled vs failing sinks through the real seam);
* ``none`` stays a strict no-op with no durable events.

All scenes are synthetic, deterministic and fast (no GPU / GUI / network).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from seestar.core import normalization_diagnostics as nd
from seestar.queuep import queue_manager as qm
from seestar.queuep.queue_manager import SeestarQueuedStacker
from seestar.core.normalization_diagnostics import (
    MAX_CHANNELS,
    MAX_RECORD_BYTES,
    STATUS_ACCEPTED,
    STATUS_NEUTRAL,
    append_record,
    artifact_filename,
    build_record,
    validate_record,
)
from seestar.core.overlap_normalization import REASON_ACCEPTED

HEADER = fits.Header()


# ---------------------------------------------------------------------------
# Minimal plain-classic stack stub (mirrors the HSI / Phase-1 harness attrs;
# the seam under test only reads these) + an output folder for the durable
# artifact.
# ---------------------------------------------------------------------------
def make_stack(norm="none", ref=None, out=None, headers=None):
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
    o._p1_tls = __import__("threading").local()
    o._p1_norm_diagnostics = []
    o._norm_reference = None
    o._norm_reference_content = None
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        o.output_folder = str(out)
    if ref is not None:
        o._capture_normalization_reference(ref)
        o._norm_reference_content = np.ones(np.asarray(ref).shape[:2], dtype=bool)
    return o


def identity_carrier(img):
    H, W = np.asarray(img).shape[:2]
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    return (M, np.ones((H, W), dtype=bool), (H, W))


def batch_item(img, carrier, srcfile="frame.fits"):
    """6-tuple batch item exactly as the plain-classic worker appends it.

    ``srcfile`` is recorded on the header under ``_SRCFILE`` exactly like
    ``_process_file`` does (value + comment), so the seam can attribute the
    frame to its real original FITS basename.
    """
    hdr = fits.Header()
    hdr["_SRCFILE"] = (srcfile, "Original source filename")
    return (
        np.array(img, dtype=np.float32, copy=True),
        hdr,
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


def _artifact_files(out):
    return sorted(Path(out).glob(f"{nd.ARTIFACT_PREFIX}_*.jsonl"))


def _read_artifact(out):
    files = _artifact_files(out)
    assert files, f"no normalization artifact in {out}"
    rows = []
    for fp in files:
        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# A1. Real FITS identity in accepted / neutral durable events
# ---------------------------------------------------------------------------
def test_accepted_sky_mean_durable_event_real_srcfile(tmp_path):
    D = _dataset()
    A, S = D["A"], D["Bs"]
    stack = make_stack("sky_mean", ref=A, out=tmp_path)
    V, _hdr, _W = stack._stack_batch(
        [
            batch_item(A, identity_carrier(A), srcfile="m31 frame 001 sky.fits"),
            batch_item(S, identity_carrier(S), srcfile="m31 frame 002 sky.fits"),
        ],
        1,
        1,
    )
    assert V is not None
    rows = _read_artifact(tmp_path)
    assert len(rows) == 2
    frames = [r["frame"] for r in rows]
    assert frames == ["m31 frame 001 sky.fits", "m31 frame 002 sky.fits"]
    for r in rows:
        assert r["normalization_method"] == "sky_mean"
        assert r["status"] == STATUS_ACCEPTED
        assert r["frame_evidence"] == nd.EVIDENCE_HEADER_SRCFILE
        assert isinstance(r["sky_offset"], (int, float))
        assert r["reason"] == REASON_ACCEPTED
        assert r["neutral_reason"] is None
        # bounded scalar contract: no nested payload / no masks
        assert set(r) <= set(nd.ALLOWED_KEYS)
        assert r["linear_a"] is None and r["linear_b"] is None
        assert 0.0 <= r["overlap_fraction"] <= 1.0
        assert r["overlap_pixel_count"] == r["effective_support_pixel_count"]


def test_accepted_linear_fit_durable_event_real_srcfile_rgb(tmp_path):
    H = W = 64
    rng = np.random.default_rng(11)
    ii = np.arange(H, dtype=np.float64)[:, None] / (H - 1)
    jj = np.arange(W, dtype=np.float64)[None, :] / (W - 1)
    base = (100.0 + 300.0 * ii + 80.0 * jj).astype(np.float32)
    A = np.stack([base, base * 0.8 + 20, base * 1.2 - 10], axis=-1).astype(np.float32)
    A += rng.normal(0.0, 2.0, size=A.shape).astype(np.float32)
    B = (1.5 * A + 40.0).astype(np.float32)
    stack = make_stack("linear_fit", ref=A, out=tmp_path)
    V, _hdr, _W = stack._stack_batch(
        [
            batch_item(A, identity_carrier(A), srcfile="rgb A frame.fits"),
            batch_item(B, identity_carrier(B), srcfile="rgb B frame.fits"),
        ],
        1,
        1,
    )
    assert V is not None
    # linear_fit maps the source ONTO the reference: B=1.5A+40 -> a=1/1.5, b=-40/1.5
    assert np.allclose(V, A, atol=1e-2), float(np.abs(V - A).max())
    rows = _read_artifact(tmp_path)
    assert len(rows) == 2
    linear = [r for r in rows if r["frame"] == "rgb B frame.fits"]
    assert len(linear) == 1
    r = linear[0]
    assert r["normalization_method"] == "linear_fit"
    assert r["status"] == STATUS_ACCEPTED
    assert len(r["linear_a"]) == 3 and len(r["linear_b"]) == 3  # bounded RGB
    assert all(np.isfinite(v) for v in r["linear_a"] + r["linear_b"])
    assert r["linear_a"][0] == pytest.approx(1.0 / 1.5, rel=2e-2)
    assert r["linear_b"][0] == pytest.approx(-40.0 / 1.5, rel=2e-2)
    assert r["sky_offset"] is None


def test_neutral_fallback_durable_event_real_srcfile(tmp_path):
    D = _dataset()
    A, B = D["A"], D["B"]
    stack = make_stack("linear_fit", ref=A, out=tmp_path)
    # Legacy 5-tuple item: header present (real identity) but NO carrier ->
    # neutral with a stable reason, still attributed to the real basename.
    item = batch_item(B, None, srcfile="neutral frame 07.fits")[:5]
    V, _h, _W = stack._stack_batch([item], 1, 1)
    assert np.allclose(V, B, atol=1e-6)  # neutral: unchanged
    rows = _read_artifact(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["frame"] == "neutral frame 07.fits"
    assert r["frame_evidence"] == nd.EVIDENCE_HEADER_SRCFILE
    assert r["status"] == STATUS_NEUTRAL
    assert r["neutral_reason"] is not None
    # Unknown geometry -> explicitly unknown, never fabricated all-valid.
    assert r["geometric_support_pixel_count"] is None
    assert r["geometric_support_fraction"] is None
    assert r["effective_support_pixel_count"] is None
    assert r["overlap_pixel_count"] is None
    # neutral linear identity vectors still bounded and recorded
    assert r["linear_a"] is not None and r["linear_b"] is not None
    assert all(v == 1.0 for v in r["linear_a"])
    assert all(v == 0.0 for v in r["linear_b"])


def test_stack_batch_attributes_survivors_after_filtered_bad_item(tmp_path):
    """A bad item (incompatible shape) is rejected BEFORE normalization; the
    surviving frames keep their real identities, in order, no drift."""
    D = _dataset()
    A, S = D["A"], D["Bs"]
    stack = make_stack("sky_mean", ref=A, out=tmp_path)
    bad_img = np.zeros((17, 17), dtype=np.float32)  # wrong shape vs 64x64
    items = [
        batch_item(A, identity_carrier(A), srcfile="first good frame.fits"),
        batch_item(bad_img, identity_carrier(bad_img), srcfile="bad frame.fits"),
        batch_item(S, identity_carrier(S), srcfile="second good frame.fits"),
    ]
    V, _h, _W = stack._stack_batch(items, 1, 1)
    assert V is not None
    rows = _read_artifact(tmp_path)
    frames = [r["frame"] for r in rows]
    assert frames == ["first good frame.fits", "second good frame.fits"]
    for r in rows:
        assert "batch_frame" not in r["frame"]
        assert r["status"] == STATUS_ACCEPTED


# ---------------------------------------------------------------------------
# A2. Append-only per-run isolation, resume/reuse does not overwrite
# ---------------------------------------------------------------------------
def test_append_within_run_and_distinct_runs_on_same_stack(tmp_path):
    D = _dataset()
    A, S = D["A"], D["Bs"]
    stack = make_stack("sky_mean", ref=A, out=tmp_path)

    # Run 1: two batches on the same capture -> ONE artifact, events appended.
    stack._stack_batch(
        [batch_item(A, identity_carrier(A), srcfile="run1 a.fits")], 1, 1
    )
    stack._stack_batch(
        [batch_item(S, identity_carrier(S), srcfile="run1 b.fits")], 1, 1
    )
    files1 = _artifact_files(tmp_path)
    assert len(files1) == 1
    rows1 = _read_artifact(tmp_path)
    assert [r["frame"] for r in rows1] == ["run1 a.fits", "run1 b.fits"]
    assert len({r["session_id"] for r in rows1}) == 1
    blob1 = files1[0].read_bytes()

    # Same stacker starts a NEW run (reference re-captured): distinct artifact,
    # prior evidence untouched (append-only, never overwritten).
    stack._release_norm_reference()
    stack._capture_normalization_reference(A)
    stack._norm_reference_content = np.ones(A.shape, dtype=bool)
    stack._stack_batch(
        [batch_item(A, identity_carrier(A), srcfile="run2 frame.fits")], 1, 1
    )
    files2 = _artifact_files(tmp_path)
    assert len(files2) == 2  # distinct session-suffixed artifact per run
    assert files1[0].read_bytes() == blob1  # run 1 evidence byte-identical
    rows2 = _read_artifact(tmp_path)
    assert len(rows2) == 3
    assert rows2[-1]["frame"] == "run2 frame.fits"
    assert rows2[-1]["session_id"] != rows1[0]["session_id"]
    # Each artifact file parses as valid JSONL and both are append-only.
    for fp in _artifact_files(tmp_path):
        for line in fp.read_text(encoding="utf-8").splitlines():
            assert json.loads(line)  # valid JSON per line


def test_none_is_strict_noop_no_durable_events(tmp_path):
    D = _dataset()
    A, B = D["A"], D["B"]
    stack = make_stack("none", ref=None, out=tmp_path)
    V, _h, _W = stack._stack_batch(
        [batch_item(B, identity_carrier(B), srcfile="none frame.fits")], 1, 1
    )
    assert np.allclose(V, B, atol=1e-6)
    assert stack._p1_norm_diagnostics == []
    assert _artifact_files(tmp_path) == []  # no durable artifact for none


# ---------------------------------------------------------------------------
# A3. Core module: allowlist, bounded records, rejection, fail-open append
# ---------------------------------------------------------------------------
def _valid_record(**over):
    kw = dict(
        frame="some frame ünïcode .fits",
        normalization_method="linear_fit",
        status=STATUS_ACCEPTED,
        reason=REASON_ACCEPTED,
        estimator="percentile_p25_p90",
        canvas_area=64 * 64,
        geometric_support_pixel_count=4096.0,
        geometric_support_fraction=1.0,
        effective_support_pixel_count=4096.0,
        effective_support_fraction=1.0,
        overlap_pixel_count=4096.0,
        overlap_fraction=1.0,
        sampled_estimator_count=1024.0,
        sky_offset=None,
        linear_a=[1.5, 1.0, 0.8],
        linear_b=[40.0, 20.0, 10.0],
        session_id="run-1",
    )
    kw.update(over)
    return build_record(**kw)


def test_core_record_preserves_unicode_and_bounds(tmp_path):
    rec = _valid_record()
    assert rec["frame"] == "some frame ünïcode .fits"  # never truncated
    ok, why = validate_record(rec)
    assert ok, why
    fp = str(tmp_path / artifact_filename("run-1"))
    assert append_record(fp, rec) is True
    lines = Path(fp).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["frame"] == "some frame ünïcode .fits"
    assert len(parsed["linear_a"]) == 3 <= MAX_CHANNELS


def test_core_record_rejects_arbitrary_arrays_and_nested_payload(tmp_path):
    fp = str(tmp_path / artifact_filename("run-x"))
    # top-level key not in the allowlist (e.g. a matrix/mask payload)
    bad = _valid_record()
    bad["M"] = np.zeros((2, 3))
    assert append_record(fp, bad) is False
    # build_record refuses non-scalar numeric fields outright (array/matrix)
    with pytest.raises(ValueError):
        _valid_record(sky_offset=np.array([[1.0]]))
    # a smuggled array on an allowlisted field is rejected by append too
    smuggled = _valid_record()
    smuggled["sky_offset"] = np.array([[1.0]])
    assert append_record(fp, smuggled) is False
    # nested dict payload
    nested = _valid_record()
    nested["reason"] = {"inner": "not scalar"}
    assert append_record(fp, nested) is False
    # channel vectors out of bounds / ragged (rejected at build AND at append)
    with pytest.raises(ValueError):
        _valid_record(linear_a=[1.0] * (MAX_CHANNELS + 1))
    ob = _valid_record()
    ob["linear_a"] = [1.0] * (MAX_CHANNELS + 1)
    assert append_record(fp, ob) is False
    with pytest.raises(ValueError):
        _valid_record(linear_a=[1.0, 2.0], linear_b=[1.0])
    ragged = _valid_record()
    ragged["linear_a"] = [1.0, 2.0]
    ragged["linear_b"] = [1.0]
    assert append_record(fp, ragged) is False
    assert not Path(fp).exists()  # nothing partially written


def test_core_record_rejects_nonfinite_and_oversize(tmp_path):
    fp = str(tmp_path / artifact_filename("run-y"))
    rec_nan = _valid_record()
    rec_nan["sky_offset"] = float("nan")
    assert append_record(fp, rec_nan) is False
    rec_inf = _valid_record()
    rec_inf["sky_offset"] = float("inf")
    assert append_record(fp, rec_inf) is False
    huge = _valid_record(frame="x" * (MAX_RECORD_BYTES + 100))
    assert append_record(fp, huge) is False
    assert not Path(fp).exists()


def test_append_fail_open_never_raises(tmp_path):
    # Denied write: target path IS an existing directory (open "a" fails).
    blocker_dir = tmp_path / "adir"
    blocker_dir.mkdir()
    assert append_record(str(blocker_dir), _valid_record()) is False
    # Directory does not exist and cannot be created (parent is a file)
    parent_file = tmp_path / "afile"
    parent_file.write_text("x", encoding="utf-8")
    assert append_record(str(parent_file / "sub" / "x.jsonl"), _valid_record()) is False
    # os.makedirs permission denied on a read-only directory (POSIX)
    import os as _os

    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o400)
    try:
        assert append_record(str(ro / "artifact.jsonl"), _valid_record()) is False
    finally:
        ro.chmod(0o700)


# ---------------------------------------------------------------------------
# A4. Fail-open through the real seam: SCI/WHT bit-identical for working vs
# disabled vs failing diagnostics sinks.
# ---------------------------------------------------------------------------
def test_seam_sink_disabled_vs_working_vs_failing_bit_identical(tmp_path, monkeypatch):
    D = _dataset()
    A, S = D["A"], D["Bs"]

    # (a) disabled: no output folder -> durable sink no-ops
    st_disabled = make_stack("sky_mean", ref=A, out=None)
    Va, _ha, Wa = st_disabled._stack_batch(
        [
            batch_item(A, identity_carrier(A), srcfile="f1.fits"),
            batch_item(S, identity_carrier(S), srcfile="f2.fits"),
        ],
        1,
        1,
    )

    # (b) working: output folder + writable artifact
    st_work = make_stack("sky_mean", ref=A, out=tmp_path)
    Vb, _hb, Wb = st_work._stack_batch(
        [
            batch_item(A, identity_carrier(A), srcfile="f1.fits"),
            batch_item(S, identity_carrier(S), srcfile="f2.fits"),
        ],
        1,
        1,
    )
    assert len(_artifact_files(tmp_path)) == 1

    # (c) failing: the append_record call raises inside the sink
    st_fail = make_stack("sky_mean", ref=A, out=tmp_path)
    real_append = nd.append_record

    def boom(*a, **k):
        raise OSError("simulated writer/serialization failure")

    monkeypatch.setattr(nd, "append_record", boom)
    try:
        Vc, _hc, Wc = st_fail._stack_batch(
            [
                batch_item(A, identity_carrier(A), srcfile="f1.fits"),
                batch_item(S, identity_carrier(S), srcfile="f2.fits"),
            ],
            1,
            1,
        )
    finally:
        monkeypatch.setattr(nd, "append_record", real_append)

    # The normalization+reducer seam output is bit-identical: diagnostics
    # (working or failing) can never change SCI/WHT.
    assert np.array_equal(Va, Vb)
    assert np.array_equal(Va, Vc)
    if Wa is not None and Wb is not None and Wc is not None:
        assert np.array_equal(Wa, Wb)
        assert np.array_equal(Wa, Wc)
    # All runs normalized (memory diagnostics present in every variant).
    assert len(st_disabled._p1_norm_diagnostics) == 2
    assert len(st_work._p1_norm_diagnostics) == 2
    assert len(st_fail._p1_norm_diagnostics) == 2


def test_seam_writer_failure_does_not_abort_run(tmp_path, monkeypatch):
    D = _dataset()
    A, B = D["A"], D["B"]
    stack = make_stack("linear_fit", ref=A, out=tmp_path)
    monkeypatch.setattr(
        nd,
        "append_record",
        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")),
    )
    # must not raise despite sink failure
    V, _h, _W = stack._stack_batch(
        [batch_item(B, identity_carrier(B), srcfile="survivor.fits")], 1, 1
    )
    assert V is not None
    assert np.allclose(V, A, atol=5e-3)  # science still applied


# ---------------------------------------------------------------------------
# A5. Count/fraction semantics (documented denominators) + in-memory bound
# ---------------------------------------------------------------------------
def test_support_count_semantics_through_seam(tmp_path):
    H = W = 32
    A = np.full((H, W), 10.0, dtype=np.float32)
    S = np.full((H, W), 13.0, dtype=np.float32)
    stack = make_stack("sky_mean", ref=A, out=tmp_path)
    stack._stack_batch(
        [
            batch_item(A, identity_carrier(A), srcfile="s1.fits"),
            batch_item(S, identity_carrier(S), srcfile="s2.fits"),
        ],
        1,
        1,
    )
    rows = _read_artifact(tmp_path)
    area = H * W
    for r in rows:
        # geometry (identity M, eroded by 1px) is known and within canvas
        assert r["canvas_area"] == area
        assert 0 < r["geometric_support_pixel_count"] <= area
        assert r["geometric_support_pixel_count"] >= r["effective_support_pixel_count"]
        assert r["sampled_estimator_count"] <= r["overlap_pixel_count"]
        # fractions are bounded [0, 1] with documented denominators
        assert 0.0 <= r["geometric_support_fraction"] <= 1.0
        assert 0.0 <= r["effective_support_fraction"] <= 1.0
        assert 0.0 <= r["overlap_fraction"] <= 1.0


def test_estimator_diag_attaches_geometric_count_while_mask_alive():
    from seestar.core.overlap_normalization import (
        estimate_sky_mean_from_geometry,
        estimate_linear_fit_from_geometry,
    )

    H = W = 64
    A = np.full((H, W), 10.0, dtype=np.float32)
    S = np.full((H, W), 13.0, dtype=np.float32)
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    content = np.ones((H, W), dtype=bool)
    _off, diag_sky = estimate_sky_mean_from_geometry(
        S, A, (H, W), M, src_content_mask_01=content, ref_content_mask=content
    )
    assert isinstance(diag_sky.get("n_geometric"), int)
    assert diag_sky["n_geometric"] > 0
    (_a, _b), diag_lin = estimate_linear_fit_from_geometry(
        S, A, (H, W), M, src_content_mask_01=content, ref_content_mask=content
    )
    assert isinstance(diag_lin.get("n_geometric"), int)
    assert diag_lin["n_geometric"] > 0
    # Missing geometry -> explicitly unknown (None), never fabricated 0/all-valid
    _off, diag_none = estimate_sky_mean_from_geometry(
        S, A, (H, W), None, src_content_mask_01=content, ref_content_mask=content
    )
    assert diag_none["n_geometric"] is None


def test_in_memory_diagnostics_bounded_ring_truthful_summary():
    stack = make_stack("sky_mean", ref=np.zeros((16, 16), dtype=np.float32))
    stack._p1_mem_diag_max_rows = 8
    from seestar.queuep.queue_manager import _P1_REASON_ACCEPTED

    for i in range(40):
        reason = _P1_REASON_ACCEPTED if i % 2 == 0 else "insufficient_overlap"
        stack._p1_record_diagnostics(f"f{i}", {"reason": reason})
    rows = stack._p1_norm_diagnostics
    assert len(rows) == 8  # ring bound respected
    summary = stack._p1_diagnostics_summary()
    assert summary["frames"] == 40  # counters stay truthful past eviction
    assert summary["accepted"] == 20
    assert summary["neutral"] == 20
    stack._p1_clear_diagnostics()
    assert stack._p1_norm_diagnostics == []
    assert stack._p1_diagnostics_summary()["frames"] == 0


def test_artifact_filename_safety():
    with pytest.raises(ValueError):
        artifact_filename("../evil")
    with pytest.raises(ValueError):
        artifact_filename("")
    assert artifact_filename("r-1").startswith(nd.ARTIFACT_PREFIX)
