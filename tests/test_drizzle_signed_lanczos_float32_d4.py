"""D4 tests: signed-Lanczos drizzle output is FORCED to float32 (engine-wide).

D3.5 made the float32 FITS preserve the signed Lanczos domain, but nothing
forced float32 when a signed Lanczos kernel was selected: a Lanczos + uint16
combination still wrote a uint16 FITS and silently clipped the legitimate
negative ringing.  D4 closes that gap:

* **Engine guard** (``SeestarQueuedStacker._save_final_stack``): whenever the
  science being serialized was produced by a signed Lanczos drizzle kernel
  (``lanczos2``/``lanczos3``) and the caller (Qt/CLI/legacy — every path
  reaches this method) requested uint16, the effective dtype is FORCED to
  float32 with the visible reason ``signed_lanczos_requires_float32``.
  ``save_as_float32_requested`` stays False (what the caller asked) while
  ``save_as_float32_effective`` becomes True (what the engine wrote).  The
  resolution is a deterministic run-start fact (kernel + request are known at
  run start): the same helper feeds ``_save_final_stack``, ``RUN_EFFECTIVE``
  and the canonical drizzle ``run_config.cfg``, so they cannot drift.
* **No silent downgrade / no over-reach**: positive kernels
  (square/gaussian/point/turbo) and Classic finalizations (where the drizzle
  kernel never ran) keep the exact requested uint16 behaviour.
* **run_config.cfg**: the canonical Drizzle config carries the deterministic
  run-start facts (``stacking_mode_effective=drizzle_direct_accumulation`` +
  substitution reason, ``drizzle_kernel_requested``/``drizzle_kernel_effective``,
  ``save_as_float32_requested`` + ``save_as_float32_effective`` resolved at run
  start, and the explicit reason when Lanczos forces float32), and its
  ``full_digest`` is byte-deterministic across an identical fresh/resume
  rebuild (existing resume-digest suites stay green).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import seestar.queuep.queue_manager as qm
import seestar.run_contract as run_contract
from seestar.core.drizzle_checkpoint import (
    DrizzleCheckpointWriter,
    build_drizzle_canonical_config,
)
from seestar.core.drizzle_core import DrizzleAccumulator

# Reuse the D3.5/D3.3 real-finalizer harnesses (same real _save_final_stack).
import tests.test_drizzle_semantics_d2 as d2

FORCE_REASON = "signed_lanczos_requires_float32"
SUBSTITUTION_REASON = "classic_reducer_not_used_by_drizzle_path"

SHAPE = (8, 8)


# ---------------------------------------------------------------------------
# 1. Engine guard: forced float32 for signed Lanczos (no silent uint16 clip)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["lanczos2", "lanczos3"])
def test_engine_guard_forces_float32_for_signed_lanczos(tmp_path, kernel):
    """A non-Qt/CLI-style caller requesting Lanczos + uint16 gets a float32
    FITS with requested=False / effective=True and the explicit reason —
    never a silent uint16 clip of the signed ringing."""
    sci = d2._drizzle_signed_science(kernel=kernel)
    assert np.min(sci) < 0.0, f"{kernel} witness did not produce ringing"

    obj = d2._stacker_dummy(tmp_path, save_as_float32=False)
    # The real engine would carry the canonicalized effective kernel.
    obj.drizzle_kernel = kernel
    obj.is_mosaic_run = False
    # Non-Qt callers still pass through the real accepted-run seam, so the
    # requested token is captured exactly as asked (uint16).  (The D3.3 e2e
    # suite proves the real capture seam stores the token; here the snapshot
    # is set directly on the lightweight finalizer harness.)
    obj._run_prov_requested = {
        "stacking_mode_requested": "winsorized-sigma-clip",
        "save_as_float32_requested": False,
        "drizzle_requested": True,
        "drizzle_kernel_requested": kernel,
        "batch_size_requested": 1,
    }

    saved = d2._engine_save(obj, sci, f"_d4_{kernel}")

    # The signed Lanczos science MUST NOT reach the uint16 export: the written
    # FITS is float32 and keeps the negative ringing.
    assert saved.dtype.kind == "f" and saved.dtype.itemsize == 4, saved.dtype
    assert saved.shape == (3,) + sci.shape
    for c in range(3):
        assert np.array_equal(np.asarray(saved[c], dtype=np.float32), sci)
    assert np.min(saved) < 0.0

    # Provenance: requested stays False (what the caller asked), effective is
    # True (what the engine wrote), reason recorded — never silent.
    assert obj.save_final_as_float32 is False
    assert obj._run_prov_requested["save_as_float32_requested"] is False
    ser = obj._serialization_effective
    assert ser["save_as_float32_effective"] is True
    assert ser["output_dtype_effective"] == "float32"
    assert ser["save_as_float32_reason"] == FORCE_REASON
    assert ser["scientific_domain_written"] == "signed_float32"


@pytest.mark.parametrize("kernel", ["square", "gaussian"])
def test_engine_guard_never_forces_positive_kernels(tmp_path, kernel):
    """Positive kernels keep the exact requested uint16 behaviour (unchanged)."""
    sci = d2._drizzle_signed_science(kernel="lanczos3")  # signed content
    obj = d2._stacker_dummy(tmp_path, save_as_float32=False)
    obj.drizzle_kernel = kernel
    obj.is_mosaic_run = False

    saved = d2._engine_save(obj, sci, f"_d4u_{kernel}")
    assert saved.dtype == np.uint16
    ser = obj._serialization_effective
    assert ser["save_as_float32_effective"] is False
    assert ser["output_dtype_effective"] == "uint16"
    assert "save_as_float32_reason" not in ser


def test_engine_guard_never_forces_classic_finalization(tmp_path):
    """A Classic finalization with a leftover Lanczos kernel is NOT forced:
    the drizzle kernel never ran on that science, so no ringing exists."""
    obj = d2._stacker_dummy(tmp_path, save_as_float32=False)
    obj.drizzle_kernel = "lanczos3"  # leftover from a previous drizzle session
    obj.is_mosaic_run = False
    data = np.full((12, 12), 5.0, dtype=np.float32)
    wht = np.ones((12, 12), dtype=np.float32)
    obj.finalization_mode = qm.FINALIZATION_MODE_REPROJECT_COADD
    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_d4_classic",
        drizzle_final_sci_data=data,
        drizzle_final_wht_data=wht,
        preserve_linear_output=True,
    )
    saved = fits.getdata(obj.final_stacked_path)
    assert saved.dtype == np.uint16
    ser = obj._serialization_effective
    assert ser["save_as_float32_effective"] is False
    assert "save_as_float32_reason" not in ser


# ---------------------------------------------------------------------------
# 2. Deterministic run-start resolution (single source of truth)
# ---------------------------------------------------------------------------


def _real_skeleton():
    s = qm.SeestarQueuedStacker(batch_size=1, autotune=False)
    s.events = []
    s.update_progress = lambda message, progress=None, level=None: s.events.append(
        str(message)
    )
    return s


def test_run_start_resolution_reason_matches_engine_guard():
    s = _real_skeleton()
    s.drizzle_active_session = True
    s.finalization_mode = qm.FINALIZATION_MODE_DRIZZLE
    s.drizzle_kernel = "lanczos3"
    s.is_mosaic_run = False
    s.save_final_as_float32 = False
    assert (
        s._signed_lanczos_forced_float32_reason() == FORCE_REASON
    )
    # The same answer when the mode is passed explicitly (as _save_final_stack
    # does) — run-start and save-time cannot drift.
    assert (
        s._signed_lanczos_forced_float32_reason(
            finalization_mode=qm.FINALIZATION_MODE_DRIZZLE
        )
        == FORCE_REASON
    )

    # Requested float32 -> nothing to force.
    s.save_final_as_float32 = True
    assert s._signed_lanczos_forced_float32_reason() is None

    # Positive kernel -> nothing to force.
    s.save_final_as_float32 = False
    s.drizzle_kernel = "square"
    assert s._signed_lanczos_forced_float32_reason() is None

    # Classic finalization with a leftover Lanczos kernel -> nothing to force.
    s.drizzle_kernel = "lanczos2"
    s.finalization_mode = qm.FINALIZATION_MODE_CLASSIC_SUMW
    assert s._signed_lanczos_forced_float32_reason() is None
    assert (
        s._signed_lanczos_forced_float32_reason(
            finalization_mode=qm.FINALIZATION_MODE_CLASSIC_SUMW
        )
        is None
    )


def test_run_effective_reports_forced_float32_with_reason():
    s = _real_skeleton()
    s.drizzle_active_session = True
    s.finalization_mode = qm.FINALIZATION_MODE_DRIZZLE
    s.drizzle_kernel = "lanczos3"
    s.is_mosaic_run = False
    s.save_final_as_float32 = False
    s._run_prov_requested = {
        "stacking_mode_requested": "winsorized-sigma-clip",
        "save_as_float32_requested": False,
        "drizzle_kernel_requested": "lanczos3",
    }
    s._capture_run_provenance_requested(
        stacking_mode_requested="winsorized-sigma-clip",
        save_as_float32_requested=False,
        drizzle_kernel_requested="lanczos3",
        drizzle_requested=True,
        batch_size_requested=1,
    )
    s._emit_run_provenance_effective()
    lines = [e for e in s.events if e.startswith("RUN_EFFECTIVE ")]
    assert len(lines) == 1, s.events
    line = lines[0]
    assert "save_as_float32_effective=true" in line
    assert f"save_as_float32_reason={FORCE_REASON}" in line
    # The requested value stays in RUN_REQUEST — never rewritten as true.
    req = [e for e in s.events if e.startswith("RUN_REQUEST ")]
    assert len(req) == 1
    assert "save_as_float32_requested=false" in req[0]


def test_run_effective_never_reports_force_for_classic():
    s = _real_skeleton()
    s.drizzle_active_session = False
    s.finalization_mode = qm.FINALIZATION_MODE_CLASSIC_SUMW
    s.drizzle_kernel = "lanczos2"  # leftover, never executed
    s.save_final_as_float32 = False
    s._run_prov_requested = {
        "stacking_mode_requested": "winsorized-sigma-clip",
        "save_as_float32_requested": False,
    }
    eff = s._run_provenance_effective()
    assert "save_as_float32_effective" not in eff
    assert "save_as_float32_reason" not in eff


# ---------------------------------------------------------------------------
# 3. Canonical drizzle run_config.cfg: deterministic run-start facts
# ---------------------------------------------------------------------------


def _wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [4.5, 4.5]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-0.001, 0.001]
    w.array_shape = SHAPE
    return w


def _configure_seed_qm(qm_obj, output, inputs, kernel, save_float32):
    qm_obj.output_folder = str(output)
    qm_obj.drizzle_active_session = True
    qm_obj.is_mosaic_run = False
    qm_obj.reproject_between_batches = False
    qm_obj.reproject_coadd_final = False
    qm_obj.move_stacked = False
    qm_obj.stacked_subdir_name = "stacked"
    qm_obj.weighting_method = "none"
    qm_obj.use_quality_weighting = False
    qm_obj.weight_by_snr = True
    qm_obj.weight_by_stars = True
    qm_obj.snr_exponent = 1.0
    qm_obj.stars_exponent = 0.5
    qm_obj.min_weight = 0.01
    qm_obj.correct_hot_pixels = True
    qm_obj.hot_pixel_threshold = 3.0
    qm_obj.neighborhood_size = 5
    qm_obj.bayer_pattern = "GRBG"
    qm_obj.drizzle_scale = 1.0
    qm_obj.drizzle_kernel = kernel
    qm_obj.drizzle_pixfrac = 1.0
    qm_obj.drizzle_wht_threshold = 0.0
    qm_obj.drizzle_wht_threshold_effective = 0.0
    qm_obj.drizzle_fillval = "0.0"
    qm_obj.drizzle_group_size = 2
    qm_obj.drizzle_processing_policy = "incremental"
    qm_obj.save_final_as_float32 = bool(save_float32)
    return qm_obj


def _identity(path):
    st = os.stat(path)
    return {
        "path": os.path.normcase(str(path)),
        "name": os.path.basename(str(path)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _seed_run_config(tmp_path, kernel="square", save_float32=False):
    """Seed a real drizzle checkpoint generation so the canonical
    ``run_config.cfg`` lands on disk (production first-commit path)."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir(parents=True)
    inputs.mkdir(parents=True)
    paths = []
    for i in range(4):
        p = inputs / f"src_{i}.fit"
        hdu = fits.PrimaryHDU(np.full(SHAPE, i + 1, dtype=np.uint16))
        hdu.header["EXPTIME"] = 1.0
        hdu.writeto(p)
        paths.append(p)
    idents = [_identity(p) for p in paths]

    qm_obj = object.__new__(qm.SeestarQueuedStacker)
    _configure_seed_qm(qm_obj, output, inputs, kernel, save_float32)
    # A real accepted run would have captured the raw request spelling; seed
    # harnesses fall back to the effective kernel (identical for valid kernels).
    cfg = build_drizzle_canonical_config(
        qm_obj, product_version=qm_obj._canonical_product_version()
    )
    writer = DrizzleCheckpointWriter(
        output, qm_obj._canonical_product_version(), cfg, _wcs(), SHAPE
    )
    accs = [
        DrizzleAccumulator(SHAPE, kernel=kernel, pixfrac=1.0) for _ in range(3)
    ]
    yy, xx = np.indices(SHAPE, dtype=np.float64)
    for i in range(2):
        data = (i * 3.0 + xx * 0.25 + yy * 0.5).astype(np.float32)
        weight = np.full(SHAPE, 0.7 + i * 0.05, dtype=np.float32)
        pixmap = np.dstack((xx + i * 0.07, yy - i * 0.04))
        for acc in accs:
            acc.add(
                data,
                weight,
                pixmap,
                exptime=1.0,
                in_units="counts",
                in_grid_mask=np.ones(SHAPE, dtype=bool),
            )
    writer.commit(
        accs,
        session_binding={
            "input_roots": [str(inputs)],
            "reference": idents[0],
            "plan": {
                "sources": idents,
                "decomposition": [4],
            },
        },
        counters={
            "frame_count": 2,
            "stacked_batches_count": 2,
            "total_exposure_seconds": 2.0,
            "exposure_unknown_count": 0,
            "exposure_min": 1.0,
            "exposure_max": 1.0,
        },
        completed_sources=idents[:2],
    )
    return output / "run_config.cfg"


@pytest.mark.parametrize("kernel", ["square", "lanczos2", "lanczos3"])
def test_canonical_cfg_carries_run_start_facts(tmp_path, kernel):
    """The canonical Drizzle run_config.cfg records the deterministic
    run-start facts (stacking substitution, kernel requested/effective, and
    the run-start save-dtype resolution: Lanczos -> float32)."""
    cfg_path = _seed_run_config(tmp_path, kernel=kernel, save_float32=False)
    assert cfg_path.is_file()
    report = run_contract.read_cfg(str(cfg_path))
    cfg = report.config

    sci = cfg.scientific
    assert sci["stacking_mode_effective"] == "drizzle_direct_accumulation"
    assert (
        sci["stacking_mode_substitution_reason"]
        == "classic_reducer_not_used_by_drizzle_path"
    )
    assert sci["drizzle_kernel_requested"] == kernel
    assert sci["drizzle_kernel_effective"] == kernel

    ex = cfg.execution
    assert ex["save_as_float32_requested"] is False
    forced = kernel in ("lanczos2", "lanczos3")
    assert ex["save_as_float32_effective"] is forced
    if forced:
        assert ex["save_as_float32_reason"] == FORCE_REASON
    else:
        assert "save_as_float32_reason" not in ex


def test_canonical_cfg_requested_float32_recorded_without_reason(tmp_path):
    """Lanczos + a requested float32 export: effective True with NO forced
    reason (the caller already asked for float32 — nothing to canonicalize)."""
    cfg_path = _seed_run_config(
        tmp_path, kernel="lanczos3", save_float32=True
    )
    cfg = run_contract.read_cfg(str(cfg_path)).config
    assert cfg.execution["save_as_float32_requested"] is True
    assert cfg.execution["save_as_float32_effective"] is True
    assert "save_as_float32_reason" not in cfg.execution


def test_canonical_cfg_digest_deterministic_across_identical_rebuild():
    """The canonical cfg is a deterministic function of the run-start state:
    two identically-configured engines (fresh write vs resume validation
    rebuild) produce byte-identical ``full_digest`` — the existing resume
    digest-equality contract cannot break — while a different kernel request
    yields a different digest."""

    def _build_cfg(kernel, save_float32):
        qm_obj = object.__new__(qm.SeestarQueuedStacker)
        _configure_seed_qm(
            qm_obj, "/tmp/out", "/tmp/in", kernel, save_float32
        )
        return build_drizzle_canonical_config(
            qm_obj, product_version=qm_obj._canonical_product_version()
        )

    a1 = _build_cfg("lanczos3", False)
    a2 = _build_cfg("lanczos3", False)
    assert a1.full_digest() == a2.full_digest()

    b1 = _build_cfg("square", False)
    b2 = _build_cfg("square", False)
    assert b1.full_digest() == b2.full_digest()

    # Different kernels / requests are distinguishable (digests differ).
    assert a1.full_digest() != b1.full_digest()
    c = _build_cfg("lanczos3", True)
    assert c.full_digest() != a1.full_digest()


def test_save_float32_reason_survives_cfg_round_trip(tmp_path):
    """The explicit reason is a registered canonical field: it round-trips
    through the real cfg reader and never lands in the unknown-keys report."""
    cfg_path = _seed_run_config(tmp_path, kernel="lanczos2", save_float32=False)
    report = run_contract.read_cfg(str(cfg_path))
    assert report.config.execution["save_as_float32_reason"] == FORCE_REASON
    assert report.unknown_keys == ()
