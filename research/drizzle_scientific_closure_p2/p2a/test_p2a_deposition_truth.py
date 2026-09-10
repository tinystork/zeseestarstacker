"""Narrow tests for the P2-A delivered-truth harness (hermetic, bounded).

These tests exercise the *delivered engine-level* extraction on a tiny
synthetic run directory.  They do not depend on the machine-local physical
artifacts and do not run the (heavier) geometry replay.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "p2a_deposition_truth", HERE / "p2a_deposition_truth.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def synthetic_run(tmp_path):
    """A minimal but structurally faithful run directory."""
    run = tmp_path / "L2"
    run.mkdir()
    ck_dir = run / ".m3d_checkpoint"
    ck_dir.mkdir()
    shape = (64, 48)
    rng = np.random.default_rng(7)
    out_img = rng.normal(0.0, 1.0, shape).astype(np.float32)
    out_img[10, 5] = 1234.5           # a "catastrophic" sample
    out_img[30, 20] = 0.75            # a "stable" sample
    out_wht = rng.normal(20.0, 0.5, shape).astype(np.float32)
    out_wht[10, 5] = 1.0e-5           # tiny positive denominator
    out_wht[30, 20] = 92.5
    np.save(ck_dir / "ch0-out_img.npy", out_img)
    np.save(ck_dir / "ch0-out_wht.npy", out_wht)
    (ck_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "output_shape_hw": list(shape),
                "total_exposure_seconds": 200.0,
                "frame_count": 10,
                "scientific_config": {"drizzle_scale_effective": 3.0},
                "completed_sources": [],
                "channels": [
                    {
                        "channel": 0,
                        "out_img": {"file": "ch0-out_img.npy"},
                        "out_wht": {"file": "ch0-out_wht.npy"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    diag = {
        "kernel": "lanczos2",
        "crop": {"height": shape[0], "width": shape[1], "x0": 0, "y0": 0},
        "conditioning_candidates": {
            "per_channel": [
                {
                    "extrema": [
                        {"kind": "sci_abs_min", "row": 10, "col": 5, "sci": float(out_img[10, 5]),
                         "wht": float(out_wht[10, 5]), "n_eff": 2.7,
                         "wht_over_local_ref": 1e-5, "wht_over_global_ref": 1e-5,
                         "wht_over_sup_w1_ref": 3e-4},
                        {"kind": "sci_abs_max", "row": 30, "col": 20, "sci": float(out_img[30, 20]),
                         "wht": float(out_wht[30, 20]), "n_eff": 2.9,
                         "wht_over_local_ref": 0.9, "wht_over_global_ref": 0.9,
                         "wht_over_sup_w1_ref": 0.8},
                        {"kind": "sci_robust_high", "row": 30, "col": 20, "sci": float(out_img[30, 20]),
                         "wht": float(out_wht[30, 20]), "n_eff": 2.9,
                         "wht_over_local_ref": 0.9, "wht_over_global_ref": 0.9,
                         "wht_over_sup_w1_ref": 0.8},
                    ]
                }
            ]
        },
    }
    (run / "drizzle_science_diagnostics.json").write_text(json.dumps(diag), encoding="utf-8")
    return run


def test_sci_identity_from_delivered_buffers(synthetic_run):
    mod = _load_mod()
    row = mod.extract_delivered_truth("L2", str(synthetic_run), "lanczos2", "cat")
    sample = next(s for s in row["samples"] if s["row"] == 10)
    assert sample["signed_native_wht_denominator"] == pytest.approx(1.0e-5, rel=1e-6)
    assert sample["signed_numerator_contribution_sum"] == pytest.approx(
        sample["delivered_sci_checkpoint_out_img"] * sample["signed_native_wht_denominator"], rel=1e-6
    )
    assert sample["reconstructed_normalized_sci"] == pytest.approx(
        sample["delivered_sci_checkpoint_out_img"], rel=1e-4
    )
    assert sample["sci_identity_rel_err"] <= 1e-4


def test_catastrophic_vs_stable_denominator(synthetic_run):
    mod = _load_mod()
    row = mod.extract_delivered_truth("L2", str(synthetic_run), "lanczos2", "cat")
    by_row = {s["row"]: s for s in row["samples"]}
    cat = by_row[10]
    stable = by_row[30]
    assert abs(cat["signed_native_wht_denominator"]) < 1e-3
    assert abs(stable["signed_native_wht_denominator"]) > 1.0
    # denominator-controlled amplification at a fixed numerator scale
    assert abs(cat["reconstructed_normalized_sci"]) > 100.0 * abs(stable["reconstructed_normalized_sci"])


def test_artifact_is_deterministic_and_schema_stable(synthetic_run, tmp_path, monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod, "OUT_ROOT", str(tmp_path))
    monkeypatch.setattr(mod, "RUNS", {"L2": ("lanczos2", "cat")})
    monkeypatch.setattr(mod, "ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(mod, "ARTIFACT_PATH", str(tmp_path / "artifacts" / "a.json"))
    # avoid the heavy replay in the determinism test
    monkeypatch.setattr(mod, "geometry_gate", lambda *a, **k: {"skipped": True})
    a1 = mod.main()
    a2 = mod.main()
    assert a1 == a2
    assert a1["schema_version"] == mod.SCHEMA_VERSION
    assert a1["phase"] == "P2-A"
    assert (tmp_path / "artifacts" / "a.json").is_file()
