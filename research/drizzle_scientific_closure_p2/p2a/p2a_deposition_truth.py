"""ZSSS Drizzle Scientific Closure — Phase P2-A bounded research harness.

Mission: ``zsss-drizzle-scientific-closure-p2-20260910`` (phase P2-A,
"direct cancellation proof").  Research only — this module is *passive*
with respect to production science.  It never mutates a delivered artifact,
never selects a threshold, never changes production code, and never declares
a conditioning rule.

What this harness does (all deterministic, bounded):

1. **Delivered engine-level decomposition facts.**  It reads the frozen
   physical artifacts of the completed runs under
   ``/home/tristan/Téléchargements/out`` (``drizzle_science_diagnostics.json``,
   the ``.m3d_checkpoint`` native ``out_img``/``out_wht`` buffers and the
   delivered final FITS) and, for a small curated sample of physical pixels,
   records:

   * the *delivered* normalized SCI (native weighted mean) — the engine's own
     ``out_img`` buffer, which is what the final FITS stores;
   * the *signed native WHT denominator sum* ``D = out_wht`` (data-independent,
     channel-invariant, exactly the sum of signed kernel contributions the
     engine accumulated);
   * the *signed numerator contribution sum* ``N = SCI * D`` (the accumulation
     identity the engine satisfies: ``out_img`` is the mean
     ``N / D``, so ``N = out_img * out_wht``);
   * the reconstructed normalized SCI ``N / D`` and its relative error against
     the delivered value;
   * the Phase-1 normalized-WHT proxies (``wht_over_local_ref``,
     ``wht_over_global_ref``, ``wht_over_sup_w1_ref``, ``n_eff``).

   These are *observed facts from the delivered buffers*, not an independent
   per-contribution replay.

2. **Frozen-geometry replay attempt (bounded).**  It reconstructs the run
   geometry from the frozen artifacts (per-frame affine from
   ``registration_diagnostics.jsonl``, output grid from the checkpoint WCS,
   ZSSS ``pixmap_from_alignment`` convention) and re-deposits the same frames
   through the *installed* ``drizzle`` 2.2.0 engine, then quantifies how well
   the reconstructed **native WHT** reproduces the delivered native WHT.
   This is the P2-A gate: a trustworthy per-contribution signed-weight
   decomposition (positive/negative/absolute sums, direct cancellation metric)
   is only meaningful if the replay reproduces the delivered native WHT.

The artifact written by :func:`main` records both blocks.  Status is left for
the architect: if the replay does not reproduce the delivered native WHT, the
per-contribution decomposition is **not** claimed (see the durable report).

Run::

    .venv/bin/python research/drizzle_scientific_closure_p2/p2a/p2a_deposition_truth.py
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Frozen physical artifacts (read-only)
# ---------------------------------------------------------------------------

OUT_ROOT = os.environ.get("ZSSS_P2A_OUT_ROOT", "/home/tristan/Téléchargements/out")
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "p2a_deposition_truth.json")

# Run directory -> (kernel label, category label)
RUNS = {
    "L2": ("lanczos2", "catastrophic_lanczos2"),
    "L3": ("lanczos3", "catastrophic_lanczos3"),
    "stop-r4": ("lanczos2", "catastrophic_lanczos2_stop"),
    "square": ("square", "benign_square"),
}

SCHEMA_VERSION = "p2a.1"

# Float32 relative tolerance for the engine accumulation identity
# SCI == (SCI * WHT) / WHT.  Pure float32 round-off plus the single
# multiply/divide; 1e-4 is a deliberately generous documented bound.
SCI_IDENTITY_RTOL = 1e-4


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _checkpoint_dir(run_dir: str) -> str:
    return os.path.join(run_dir, ".m3d_checkpoint")


def _load_native(run_dir: str, ch: int):
    ck = _load_json(os.path.join(_checkpoint_dir(run_dir), "checkpoint.json"))
    entry = ck["channels"][ch]
    oi = np.load(os.path.join(_checkpoint_dir(run_dir), entry["out_img"]["file"]))
    ow = np.load(os.path.join(_checkpoint_dir(run_dir), entry["out_wht"]["file"]))
    return ck, oi, ow


def _delivered_fits(run_dir: str, ch: int):
    from astropy.io import fits

    cands = sorted(glob.glob(os.path.join(run_dir, "*.fit")))
    if not cands:
        return None
    return np.asarray(fits.open(cands[0])[0].data[ch], dtype=np.float64)


def _sample_coordinates(diag: dict, ch: int = 0, per_side: int = 3):
    """Curated physical coordinates for one channel.

    Always includes the exact absolute-SCI extrema and the largest-native-WHT
    ``sci_robust_high`` sample (the stable control), then the next
    ``per_side - 1`` smallest/largest robust values.
    """
    cond = diag["conditioning_candidates"]["per_channel"][ch]
    ex = cond["extrema"]
    picks: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def add(e):
        key = (int(e["row"]), int(e["col"]))
        if key in seen:
            return
        seen.add(key)
        picks.append(e)

    for kind in ("sci_abs_min", "sci_abs_max"):
        for e in ex:
            if e["kind"] == kind:
                add(e)
                break
    # stable control: the robust-high sample with the largest native WHT
    highs = [e for e in ex if e["kind"] == "sci_robust_high"]
    if highs:
        add(max(highs, key=lambda e: abs(float(e["wht"]))))
        for e in sorted(highs, key=lambda e: -float(e["wht"]))[: max(0, per_side - 1)]:
            add(e)
    lows = [e for e in ex if e["kind"] == "sci_robust_low"]
    for e in sorted(lows, key=lambda e: float(e["wht"]))[:per_side]:
        add(e)
    return picks


def extract_delivered_truth(run: str, run_dir: str, kernel: str, category: str):
    diag = _load_json(os.path.join(run_dir, "drizzle_science_diagnostics.json"))
    ch = 0
    ck, oi, ow = _load_native(run_dir, ch)
    fits_sci = _delivered_fits(run_dir, ch)

    rows = []
    for e in _sample_coordinates(diag, ch):
        r, c = int(e["row"]), int(e["col"])
        sci = float(oi[r, c])              # engine native weighted mean
        wht = float(ow[r, c])              # signed native WHT denominator sum
        num = sci * wht                    # signed numerator contribution sum
        recon = num / wht if wht != 0.0 else float("nan")
        rel = abs(recon - sci) / abs(sci) if sci != 0.0 else 0.0
        rows.append(
            {
                "row": r,
                "col": c,
                "kind": e.get("kind"),
                "delivered_sci_diagnostics": float(e["sci"]),
                "delivered_sci_checkpoint_out_img": sci,
                "delivered_sci_fits": None if fits_sci is None else float(fits_sci[r, c]),
                "signed_native_wht_denominator": wht,
                "signed_numerator_contribution_sum": num,
                "reconstructed_normalized_sci": recon,
                "sci_identity_rel_err": rel,
                "n_eff": e.get("n_eff"),
                "sup_w1": e.get("sup_w1"),
                "phase1_wht_over_local_ref": e.get("wht_over_local_ref"),
                "phase1_wht_over_global_ref": e.get("wht_over_global_ref"),
                "phase1_wht_over_sup_w1_ref": e.get("wht_over_sup_w1_ref"),
                "fits_matches_checkpoint": (
                    None if fits_sci is None else bool(np.isclose(float(fits_sci[r, c]), sci, rtol=0, atol=0))
                ),
            }
        )

    return {
        "run": run,
        "kernel": kernel,
        "category": category,
        "channel": ch,
        "output_shape_hw": list(ck["output_shape_hw"]),
        "total_exptime": ck.get("total_exposure_seconds"),
        "frame_count": ck.get("frame_count"),
        "samples": rows,
    }


# ---------------------------------------------------------------------------
# frozen-geometry replay attempt (bounded)
# ---------------------------------------------------------------------------


def _reconstruct_pixmaps(run_dir: str):
    """Rebuild per-frame output-grid pixmaps from the frozen run artifacts.

    Uses the ZSSS convention exactly as ``seestar.core.drizzle_core`` documents
    it: ``tf`` maps ORIGINAL pixel ``(x, y)`` to reference-grid pixels and the
    output grid is the reference grid scaled by ``scale`` (CDELT/scale,
    CRPIX*scale), so the reference->output WCS round trip is the identity and
    ``pixmap = scale * (tf @ [x, y, 1])``.
    """
    ck = _load_json(os.path.join(_checkpoint_dir(run_dir), "checkpoint.json"))
    scale = float(ck["scientific_config"]["drizzle_scale_effective"])
    out_h, out_w = ck["output_shape_hw"]
    regs = {}
    with open(os.path.join(run_dir, "registration_diagnostics.jsonl"), "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("event") == "registration" and d.get("success"):
                regs[d["frame"]] = d
    names = [s["name"] for s in ck["completed_sources"] if s["name"] in regs]
    return ck, scale, names, regs, (out_h, out_w)


def reconstruct_native_wht(run_dir: str, kernel: str, frame_hw=(1920, 1080)):
    """Re-deposit the frozen frames through the real engine (native WHT only).

    Data is irrelevant to ``out_wht``; only the pixmap and the weight map
    matter, so this is a pure geometry replay.
    """
    from drizzle.resample import Drizzle

    ck, scale, names, regs, out_sh = _reconstruct_pixmaps(run_dir)
    H, W = frame_hw
    yy, xx = np.indices((H, W), dtype=np.float64)
    acc = np.zeros(out_sh, dtype=np.float32)
    for name in names:
        d = regs[name]
        th = np.radians(d["applied_rotation_deg"])
        tx, ty = d["applied_translation"]
        co, si = np.cos(th), np.sin(th)
        px = scale * (co * xx - si * yy + tx)
        py = scale * (si * xx + co * yy + ty)
        in_grid = ((px >= 0) & (px < out_sh[1]) & (py >= 0) & (py < out_sh[0])).astype(np.float32)
        eng = Drizzle(
            out_img=np.zeros(out_sh, dtype=np.float32),
            out_wht=np.zeros(out_sh, dtype=np.float32),
            kernel=kernel,
            fillval="0.0",
        )
        eng.add_image(
            data=np.zeros((H, W), dtype=np.float32),
            exptime=20.0,
            pixmap=np.dstack((px, py)),
            weight_map=in_grid,
            in_units="counts",
            pixfrac=1.0,
            wht_scale=20.0,
        )
        acc += np.array(eng.out_wht, dtype=np.float32)
    return acc, out_sh


def geometry_gate(run_dir: str, kernel: str, stride: int = 15) -> dict:
    """Quantify whether the frozen-geometry replay reproduces the delivered WHT."""
    ck, oi, ow = _load_native(run_dir, 0)
    recon, _ = reconstruct_native_wht(run_dir, kernel)
    a = ow[::stride, ::stride].astype(np.float64).ravel()
    b = recon[::stride, ::stride].astype(np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    corr = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else None

    # best integer 2D shift of the reconstruction against the delivered WHT
    from numpy.fft import irfft2, rfft2

    best = {"corr": -2.0, "dy": None, "dx": None}
    A = ow - np.mean(ow)
    B = recon.astype(np.float64) - np.mean(recon)
    cc = irfft2(rfft2(A) * np.conj(rfft2(B)), s=A.shape)
    idx = np.unravel_index(np.argmax(cc), cc.shape)
    dy, dx = int(idx[0]), int(idx[1])
    if dy > A.shape[0] // 2:
        dy -= A.shape[0]
    if dx > A.shape[1] // 2:
        dx -= A.shape[1]
    shifted = np.roll(np.roll(recon, dy, 0), dx, 1)
    aa = ow[::stride, ::stride].astype(np.float64).ravel()
    bb = shifted[::stride, ::stride].astype(np.float64).ravel()
    best = {"corr": float(np.corrcoef(aa, bb)[0, 1]), "dy": dy, "dx": dx}
    return {
        "corr_zero_shift": corr,
        "max_abs_diff": float(np.abs(ow - recon).max()),
        "best_integer_shift": best,
        "delivered_median": float(np.median(ow)),
        "recon_median": float(np.median(recon)),
        "recon_negative_fraction": float(np.mean(recon < 0)),
        "delivered_negative_fraction": float(np.mean(ow < 0)),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build_artifact() -> dict:
    runs: list[dict] = []
    gates: dict[str, dict] = {}
    for run, (kernel, category) in RUNS.items():
        run_dir = os.path.join(OUT_ROOT, run)
        if not os.path.isdir(run_dir):
            continue
        runs.append(extract_delivered_truth(run, run_dir, kernel, category))
        gates[run] = geometry_gate(run_dir, kernel)
    return {
        "schema_version": SCHEMA_VERSION,
        "mission_id": "zsss-drizzle-scientific-closure-p2-20260910",
        "phase": "P2-A",
        "diagnostic_only": True,
        "sci_identity_rtol": SCI_IDENTITY_RTOL,
        "out_root": OUT_ROOT,
        "runs": runs,
        "geometry_gate": gates,
    }


def main() -> dict:
    art = build_artifact()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=1, sort_keys=True)
    print(f"wrote {ARTIFACT_PATH}")
    for r in art["runs"]:
        worst = max((s["sci_identity_rel_err"] for s in r["samples"]), default=0.0)
        print(f"  {r['run']:>8s} kernel={r['kernel']:<9s} samples={len(r['samples']):2d} "
              f"max_sci_identity_rel_err={worst:.3e}")
    for run, g in art["geometry_gate"].items():
        if "corr_zero_shift" not in g:
            print(f"  GATE {run:>8s} skipped")
            continue
        shift = g["best_integer_shift"]
        print(f"  GATE {run:>8s} corr0={g['corr_zero_shift']:.4f} "
              f"best_shift=({shift['dy']},{shift['dx']}) "
              f"best_corr={shift['corr']:.4f}")
    return art


if __name__ == "__main__":
    main()
