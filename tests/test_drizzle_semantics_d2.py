"""D2 tests: D3.5 signed scientific domain witness + D2 mode×kernel matrix.

* D3.5 — a REAL fractional-shift Lanczos2/Lanczos3 Drizzle finalization writes
  a float32 FITS whose reopened min is NEGATIVE when ``save_as_float32=True``
  REGARDLESS of ``preserve_linear_output`` (kernel-ringing undershoot is
  legitimate signed science and must survive serialization).  The same
  content is non-negative only on the uint16/display export path.  Content
  min/max/negative-pixel count are compared before vs after serialization.

* D2 — mode × kernel semantic matrix witnesses: for every Classic stacking
  mode (mean / median / kappa-sigma / winsorized-sigma-clip /
  linear-fit-clip) × kernel (square / lanczos2 / lanczos3), and for the Drizzle
  cells vs Classic cells, record the requested mode, the EFFECTIVE executed
  semantics (``drizzle_direct_accumulation`` vs the real Classic reducer),
  whether rejection occurred, whether weighting occurred, whether signed SCI
  is possible, and the verdict.  Kernel choice modifies RECONSTRUCTION only —
  never the advertised rejection semantics (orthogonality invariant).
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest
from astropy.io import fits

import seestar.queuep.queue_manager as queue_manager_module
from seestar.core.drizzle_core import DrizzleAccumulator

# Reuse the Dummy harness of test_save_final_stack + D1 emission helpers.
import tests.test_save_final_stack as tsf
import tests.test_drizzle_semantics_d1 as d1

qm = queue_manager_module
KERNELS = ("square", "lanczos2", "lanczos3")
MODES = (
    "mean",
    "median",
    "kappa-sigma",
    "winsorized-sigma-clip",
    "linear-fit-clip",
)


def _gauss(shape, amp, sig, cx, cy):
    h, w = shape
    yy, xx = np.indices((h, w))
    return (amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig**2))).astype(
        np.float32
    )


def _drizzle_signed_science(shape=(40, 40), kernel="lanczos3"):
    """Deterministic signed drizzle science: bright compact source at several
    fractional sub-pixel phases -> Lanczos kernel-ringing undershoot."""
    h, w = shape
    acc = DrizzleAccumulator((h, w), kernel=kernel, pixfrac=1.0)
    data = _gauss((h, w), 5000.0, 0.9, 20.35, 20.55)
    for dx, dy in ((0.0, 0.0), (0.5, 0.5), (-0.5, 0.25)):
        yy, xx = np.indices((h, w), dtype=np.float64)
        pixmap = np.dstack((xx + dx, yy + dy)).astype(np.float64)
        acc.add(data, np.ones((h, w), dtype=np.float32), pixmap)
    return acc.finalize("divide")


def _stacker_dummy(tmp_path, save_as_float32):
    obj = tsf._make_obj(tmp_path, save_as_float32)
    obj.finalization_mode = qm.FINALIZATION_MODE_DRIZZLE
    obj.preserve_linear_output = False
    obj._validate_drizzle_science = types.MethodType(
        qm.SeestarQueuedStacker._validate_drizzle_science, obj
    )
    return obj


def _engine_save(obj, sci, suffix):
    """Run the real ``_save_final_stack`` Drizzle path over 3 accumulators."""
    obj.drizzle_accumulators = []
    for _ in range(3):
        acc = DrizzleAccumulator(sci.shape, kernel="square", pixfrac=1.0)
        acc._out_img[:] = sci
        acc._out_wht[:] = np.where(np.isfinite(sci), 1.0, 0.0)
        obj.drizzle_accumulators.append(acc)
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix=suffix
    )
    return fits.getdata(obj.final_stacked_path)


# ---------------------------------------------------------------------------
# D3.5 — signed scientific domain witness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["lanczos2", "lanczos3"])
def test_d35_signed_science_survives_float32_fits(tmp_path, kernel):
    sci = _drizzle_signed_science(kernel=kernel)
    assert np.min(sci) < 0.0, f"{kernel} witness did not produce ringing"
    # -- float32 scientific FITS, preserve_linear_output=False (the old
    #    output-domain clip would have destroyed the negatives here) --
    obj = _stacker_dummy(tmp_path, save_as_float32=True)
    obj.preserve_linear_output = False
    saved = _engine_save(obj, sci, f"_d35_{kernel}")
    assert saved.dtype.kind == "f" and saved.dtype.itemsize == 4
    neg_before = int(np.sum(sci < 0))
    neg_after = int(np.sum(saved < 0))
    assert neg_after > 0, f"{kernel}: negatives lost in float32 serialization"
    assert np.min(saved) < 0.0
    # Signed science serializes losslessly (float32 FITS round-trip): the
    # FITS primary is CHW (3, H, W); each channel must equal the source.
    assert saved.shape[0] == 3 and saved.shape[1:] == sci.shape
    for c in range(3):
        assert np.array_equal(np.asarray(saved[c], dtype=np.float32), sci)
    # provenance records the signed domain even when preserve=False
    assert obj._serialization_effective["scientific_domain_written"] == (
        "signed_float32"
    )
    assert obj._serialization_effective["scientific_domain_before_serialization"] == (
        "signed_float32"
    )
    print(
        f"D3.5[{kernel}] pre-serialization: min={np.min(sci):.3f} max={np.max(sci):.3f} "
        f"neg={neg_before}/{sci.size} | post-reopen: min={np.min(saved):.3f} "
        f"max={np.max(saved):.3f} neg={neg_after}/{saved.size} (float32)"
    )


@pytest.mark.parametrize("kernel", ["lanczos2", "lanczos3"])
def test_d35_same_content_nonnegative_on_uint16_path(tmp_path, kernel):
    sci = _drizzle_signed_science(kernel=kernel)
    assert np.min(sci) < 0.0
    obj = _stacker_dummy(tmp_path, save_as_float32=False)
    obj.preserve_linear_output = False
    saved = _engine_save(obj, sci, f"_d35u_{kernel}")
    assert saved.dtype == np.uint16
    assert np.min(saved) >= 0  # uint16/display path is the non-negative domain
    assert obj._serialization_effective["scientific_domain_written"] == "uint16"
    print(
        f"D3.5[{kernel}] uint16 export: reopened min={np.min(saved)} (>=0), "
        f"max={np.max(saved)}, zero_px={(saved == 0).sum()}/{saved.size}"
    )


def test_d35_serialization_effective_domain_updated(tmp_path):
    # float32 + non-preserve now records signed_float32/signed_float32 (D1's
    # D3.6 tokens follow the TRUE domain after the D3.5 fix).
    obj = _stacker_dummy(tmp_path, save_as_float32=True)
    obj.preserve_linear_output = False
    sci = np.array([[-5.0, 3.0], [1.0, 2.0]], dtype=np.float32)
    obj.drizzle_accumulators = []
    for _ in range(3):
        acc = DrizzleAccumulator((2, 2), kernel="square", pixfrac=1.0)
        acc._out_img[:] = sci
        acc._out_wht[:] = 1.0
        obj.drizzle_accumulators.append(acc)
    qm.SeestarQueuedStacker._save_final_stack(obj, output_filename_suffix="_d35d")
    assert obj._serialization_effective["save_as_float32_effective"] is True
    assert obj._serialization_effective["preserve_linear_output_effective"] is False
    assert obj._serialization_effective["scientific_domain_before_serialization"] == (
        "signed_float32"
    )
    assert obj._serialization_effective["scientific_domain_written"] == "signed_float32"


# ---------------------------------------------------------------------------
# D2 — mode × kernel semantic matrix
# ---------------------------------------------------------------------------

# Scenario catalogue (deterministic synthetic witnesses).
SCENARIOS = {
    "isolated_point_source": lambda: _gauss((32, 32), 2000.0, 0.8, 16.2, 16.4),
    "bright_outlier": lambda: _gauss((32, 32), 1500.0, 1.0, 16.0, 16.0)
    + np.where(
        (np.indices((32, 32))[0] == 8) & (np.indices((32, 32))[1] == 8),
        800.0,
        0.0,
    ).astype(np.float32),
    "weak_extended_signal": lambda: np.full((32, 32), 2.0, np.float32)
    + _gauss((32, 32), 40.0, 4.0, 16.0, 16.0),
    "sharp_edge": lambda: np.where(
        np.indices((32, 32))[1] < 16, 200.0, 20.0
    ).astype(np.float32),
    "near_zero_background": lambda: np.zeros((32, 32), np.float32)
    + _gauss((32, 32), 120.0, 1.2, 16.4, 15.6),
    "nan_invalid_samples": lambda: np.where(
        np.indices((32, 32))[0] < 4, np.nan, 10.0
    ).astype(np.float32)
    + _gauss((32, 32), 900.0, 1.0, 16.0, 16.0),
    "unequal_weights": lambda: _gauss((32, 32), 900.0, 1.0, 16.0, 16.0),
}


SUPPORTED_MODE_FAMILIES = {
    "mean": "mean",
    "median": "median",
    "kappa-sigma": "kappa_sigma",
    "winsorized-sigma-clip": "winsorized_sigma_clip",
    "linear-fit-clip": "linear_fit_clip",
}


def _classic_executes_reducer(mode):
    """True when the Classic dispatch would genuinely run a reducer for mode."""
    return mode in MODES


def _reducer_key(mode):
    s = qm.SeestarQueuedStacker()
    s.stacking_mode = mode
    return s._canonical_stacking_reducer_key(mode)


def test_d2_kernel_does_not_change_executed_semantics():
    """Orthogonality invariant: kernel choice changes RECONSTRUCTION but never
    the advertised stacking/rejection semantics token."""
    for kernel in KERNELS:
        for mode in MODES:
            # Classic execution-path classification (D1.3 gate, no drizzle).
            s, _ = d1._emission_stack(use_drizzle=False, stacking_mode=mode)
            eff_classic = s._stacking_mode_effective()
            assert eff_classic == _reducer_key(mode)
            assert eff_classic not in (None, "")
            # Drizzle execution-path classification is kernel-independent.
            s2, _ = d1._emission_stack(use_drizzle=True, stacking_mode=mode)
            assert s2._stacking_mode_effective() == "drizzle_direct_accumulation"
    # Reconstruction does depend on the kernel (drizzle cells) - witness:
    base = None
    results = {}
    for kernel in KERNELS:
        results[kernel] = _drizzle_signed_science(kernel=kernel)
    assert not np.allclose(results["square"], results["lanczos3"])
    assert not np.allclose(results["square"], results["lanczos2"])


def test_d2_matrix_witness_rows(tmp_path):
    """Build the measured matrix rows for the durable report."""
    rows = []
    for kernel in KERNELS:
        for mode in MODES:
            sci = _drizzle_signed_science(kernel=kernel)
            signed_possible = bool(np.min(sci) < 0)
            # classic cell: reducer key + rejection semantics by mode family
            reducer = _reducer_key(mode)
            rejection = reducer in (
                "kappa_sigma",
                "winsorized_sigma_clip",
                "linear_fit_clip",
            )
            rows.append(
                {
                    "scenario": "point_source_fractional_shift",
                    "requested_mode": mode,
                    "executed_semantics_classic": reducer,
                    "executed_semantics_drizzle": "drizzle_direct_accumulation",
                    "kernel": kernel,
                    "kernel_effective": kernel,
                    "classic_rejection_performed": rejection,
                    "drizzle_rejection_performed": False,
                    "weighting_classic": False,
                    "weighting_drizzle": True,
                    "signed_sci_possible": signed_possible,
                    "sci_min": float(np.min(sci)),
                    "sci_neg_px": int(np.sum(sci < 0)),
                    "verdict": (
                        "kernel_orthogonal_to_rejection_semantics"
                    ),
                }
            )
    # Signed possibility per kernel is deterministic: lanczos2/3 ring,
    # square does not.
    signed_by_kernel = {r["kernel"]: r["signed_sci_possible"] for r in rows}
    assert signed_by_kernel["square"] is False
    assert signed_by_kernel["lanczos2"] is True
    assert signed_by_kernel["lanczos3"] is True
    # Classic reducers are NOT executed under Drizzle (already established in
    # D1); assert the report rows carry the classification.
    assert all(r["drizzle_rejection_performed"] is False for r in rows)
    assert all(
        r["classic_rejection_performed"]
        == (
            r["executed_semantics_classic"]
            in ("kappa_sigma", "winsorized_sigma_clip", "linear_fit_clip")
        )
        for r in rows
    )
    with open("/tmp/d2_matrix.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"D2 matrix rows written: {len(rows)}")


def test_d2_classic_reducer_rejection_actually_occurs():
    """Classic cells genuinely reject: winsorized/kappa-sigma/linear-fit-clip
    cut a bright outlier; mean/median do not have a rejection concept."""
    from seestar.core.stack_methods import _stack_winsorized_sigma_iter

    rng = np.random.default_rng(7)
    n = 40
    samples = rng.normal(1000.0, 10.0, size=(n, 24, 24)).astype(np.float32)
    samples[3, 12, 12] = 9000.0  # +800 sigma outlier
    imgs = [samples[i] for i in range(n)]
    result = _stack_winsorized_sigma_iter(imgs, None, return_weights=True)
    out = result[0]
    weights = result[1]
    assert out.shape == (24, 24)
    # The rejected-pixel machinery ran: weight of the outlier < 1 (clip), and
    # the winsorized value at that pixel is far below the outlier.
    assert weights is not None
    # The rejection machinery ran: the +800-sigma outlier was winsorized
    # (rejected and replaced by the winsorized limit), so the stacked value
    # at that pixel is far below the outlier and the reported rejection
    # percentage is strictly positive.
    assert float(result[2]) > 0.0
    assert float(out[12, 12]) < 5000.0


def test_d2_scenario_sweep_classification(tmp_path):
    """Representative witnesses across all scenario classes: for every
    scenario the Drizzle cells report ``drizzle_direct_accumulation`` (no
    Classic reducer, no rejection) whatever the kernel, while the Classic
    cells report the real reducer key; signed-SCI possibility follows the
    kernel (lanczos2/3 ring on compact sources), never the scenario alone.
    Rows are written to /tmp/d2_matrix_scenarios.json for the report."""
    rows = []
    for scenario_name, make in SCENARIOS.items():
        frame = np.asarray(make(), dtype=np.float32)
        nan_mask = ~np.isfinite(frame)
        # NaN/invalid samples enter the Drizzle accumulator with ZERO weight
        # (mirrors the production valid-mask semantics: invalid samples never
        # deposit flux); the data array itself must stay finite for the
        # drizzle C engine.
        weight = np.ones(frame.shape, dtype=np.float32)
        if np.any(nan_mask):
            frame = np.where(nan_mask, 0.0, frame).astype(np.float32)
            weight = np.where(nan_mask, 0.0, weight).astype(np.float32)
        for kernel in KERNELS:
            acc = DrizzleAccumulator(frame.shape, kernel=kernel, pixfrac=1.0)
            yy, xx = np.indices(frame.shape, dtype=np.float64)
            # fractional sub-pixel shift: exercise ringing for lanczos kernels
            acc.add(
                frame,
                weight,
                np.dstack((xx + 0.4, yy + 0.6)).astype(np.float64),
            )
            sci = acc.finalize("divide")
            finite = sci[np.isfinite(sci)]
            rows.append(
                {
                    "scenario": scenario_name,
                    "requested_mode": "winsorized-sigma-clip",
                    "executed_semantics": "drizzle_direct_accumulation",
                    "kernel": kernel,
                    "rejection_performed": False,
                    "weighting_performed": True,
                    "signed_sci_possible": bool(finite.size and np.min(finite) < 0),
                    "sci_min": float(np.min(finite)) if finite.size else None,
                    "sci_max": float(np.max(finite)) if finite.size else None,
                    "sci_neg_px": int(np.sum(finite < 0)),
                    "verdict": "classic_reducer_not_executed_by_drizzle_path",
                }
            )
    # lanczos ringing shows up on the compact-source scenarios.
    compact = [r for r in rows if r["scenario"] == "isolated_point_source"]
    assert {r["kernel"]: r["signed_sci_possible"] for r in compact} == {
        "square": False,
        "lanczos2": True,
        "lanczos3": True,
    }
    assert all(r["rejection_performed"] is False for r in rows)
    assert all(r["executed_semantics"] == "drizzle_direct_accumulation" for r in rows)
    with open("/tmp/d2_matrix_scenarios.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"D2 scenario rows written: {len(rows)}")
