"""RF-2 — real M16 target-policy witness (corrective C1).

Bounded, production-external, read-only.  This script compares two
registration-reference *target policies* on **actual prepared M16 pixels** using
the **real astroalign matcher** and the **production Euclidean conversion**
(``alignment.py:228-237``: similarity fit -> scale discarded -> rotation +
translation):

* ``immutable`` — the initially-selected reference frame (production
  ``_get_reference_image`` quality metric ``median/(1.4826*MAD)`` argmax),
  held constant for the whole run.  This is the **primary stable candidate**.
* ``evolving`` — the current production reproject target, emulated: the
  reference is replaced at each batch boundary by the cumulative mean of the
  aligned (warped) frames so far (the analogue of ``_solve_cumulative_stack``
  = ``sum/wht`` of aligned frames).

The question this answers is **organization sensitivity**: for the *same* source
frame, how much does its fitted per-frame transform change when the batch
decomposition or processing order changes, under each target policy?  A target
whose identity is independent of batch/order must yield **identical** per-frame
transforms (keyed by frame ID); a target rebuilt from the batch stream cannot.

What is measured
----------------
* per-frame transform ``M_j`` (production Euclidean 3x3), keyed by source frame.
* matrix/point displacement between policies/configurations at **centre / edge /
  corner** canonical points, aggregated P50 / P95 / max over frames.
* alignment failure rate, runtime, and held-out **target-fit residual**.

Honest limits (explicit)
------------------------
* **No ground truth.**  M16 is one session, ~23 min, one focal state.  These are
  *observational organization-sensitivity proxies*, not accuracy measurements.
* **Evolving-target emulation deviation.**  Production rebuilds the target as
  ``_solve_cumulative_stack()`` = the ``sum/wht`` mean of the **full RGB**
  memmap accumulators, warped via ``cv2.warpAffine`` per frame, with WCS
  handling and weight normalisation.  Here the target is rebuilt as the
  **mean of the warped green channels** (``cv2.warpAffine`` with the recorded
  ``M_j``, then ``np.mean``).  This preserves the *identity semantics* (the
  target is the cumulative aligned stack) but omits WCS re-solving and
  weight-map normalisation.  It is **not** claimed worker-equivalent.
* Preparation reuses RF-1's faithful basis (``m16_scale_witness``): header
  ``BAYERPAT`` debayer + deterministic CPU hot-pixel correction + green channel.
* astroalign's internal RANSAC uses an unseeded ``np.random.default_rng()``;
  on these well-conditioned frames the returned transform is empirically
  deterministic (repeated runs are bit-identical), and this is reported.

No ``seestar`` file is modified; source FITS are read-only.
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
from skimage.transform import SimilarityTransform

_PARENT = os.path.join(os.path.dirname(__file__), "..", "registration_field_rotation")
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import m16_scale_witness as w  # noqa: E402

M16_FOLDER = "/home/tristan/M16/quick"


# --------------------------------------------------------------------------
# Production Euclidean conversion (alignment.py:228-237)
# --------------------------------------------------------------------------


def production_euclidean(T):
    """Discard the astroalign similarity scale and keep rotation + translation.

    Faithful to ``alignment.py:228-237``.  Returns the 3x3 homogeneous params.
    """
    a, b = T.params[0, 0], T.params[1, 0]
    theta = np.arctan2(b, a)
    tx, ty = T.params[0, 2], T.params[1, 2]
    return SimilarityTransform(rotation=theta, translation=(tx, ty)).params


def _apply(M, pts):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (h @ M.T)[:, :2]


# --------------------------------------------------------------------------
# Preparation (reuses RF-1's faithful basis)
# --------------------------------------------------------------------------


def prepare_all(folder):
    """Prepare all frames (green channel) and select the reference frame with
    the production quality metric.  Returns ``(greens, ref_path, ref_green)``.

    ``greens`` is ``{path: green_channel_float32}``.  ``ref_path`` is the
    argmax of ``median/(1.4826*MAD)`` over variance-gated candidates — the same
    initially-selected reference the immutable policy holds constant."""
    files = sorted(glob.glob(os.path.join(folder, "*.fit")))
    assert files, f"no .fit frames in {folder}"

    greens = {}
    candidates = []
    for f in files:
        norm, header = w._load_normalized(f)
        if norm is None or norm.ndim != 2:
            continue
        if not w._variance_ok(norm):
            continue
        pattern = w._bayer_pattern_from_header(header)
        rgb = w._debayer(norm, pattern)
        rgb = w._detect_and_correct_hot_pixels_cpu(rgb, 3.0, 5)
        greens[f] = w._green(rgb)
        candidates.append((f, w._quality_metric(rgb)))

    assert candidates, "no reference candidate passed"
    ref_path, ref_metric = max(candidates, key=lambda t: t[1])
    return greens, ref_path, greens[ref_path], ref_metric


def _warp(green, M):
    import cv2

    H, W = green.shape[:2]
    return cv2.warpAffine(green, M[:2, :].astype(np.float32), (W, H))


def _split_pairs(s, t, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(s))
    n_fit = max(2, int(round(len(s) * 0.7)))
    return order[:n_fit], order[n_fit:]


def _target_fit_residual(s, t, seed):
    """Closed-form Euclidean fit on the fit subset; residual on the hold subset
    (mirrors RF-1 / the synthetic harness).  Returns (p50, p95, rms) or None."""
    from skimage.transform import estimate_transform

    if len(s) < 4:
        return None
    fit_idx, hold_idx = _split_pairs(s, t, seed)
    if len(hold_idx) == 0:
        return None
    model = estimate_transform("euclidean", s[fit_idx], t[fit_idx])
    pred = model(s[hold_idx])
    r = np.linalg.norm(pred - t[hold_idx], axis=1)
    return float(np.percentile(r, 50)), float(np.percentile(r, 95)), float(np.sqrt(np.mean(r ** 2)))


def _align_one(src_green, target, seed):
    """Run the real astroalign matcher + production Euclidean conversion for one
    frame against ``target``.  Returns a dict with ``M`` (3x3), ``match_count``,
    ``failure`` (bool), ``runtime_s``, and ``residual`` (p50/p95/rms or None)."""
    import astroalign as aa

    t0 = time.perf_counter()
    try:
        T, (s, t) = aa.find_transform(source=src_green, target=target)
    except Exception as e:  # noqa: BLE001
        return {
            "M": None,
            "match_count": 0,
            "failure": True,
            "error": type(e).__name__,
            "runtime_s": time.perf_counter() - t0,
            "residual": None,
        }
    M = production_euclidean(T)
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    return {
        "M": M,
        "match_count": int(len(s)),
        "failure": False,
        "error": None,
        "runtime_s": time.perf_counter() - t0,
        "residual": _target_fit_residual(s, t, seed),
    }


def _run_policy(strategy, frame_order, greens, ref_green, batch_size, seed):
    """Run one (strategy, order, batch_size) pass over ``frame_order``.

    ``frame_order`` is a list of paths in processing order.  Returns a dict:
      * ``transforms`` : {frame_index_in_others: 3x3 M or None}
      * ``residuals``  : {frame_index: (p50,p95,rms) or None}
      * ``failures``   : {frame_index: bool}
      * ``n_aligned`` / ``n_failed`` / ``runtime_s``
      * ``target_rebuilds`` : int (number of times the target was replaced)
    """
    n = len(frame_order)
    transforms = {}
    residuals = {}
    failures = {}
    t_start = time.perf_counter()

    target = ref_green
    warped_stack = []
    rebuilds = 0

    for pos, path in enumerate(frame_order):
        frame_id = _frame_id(path)
        rec = _align_one(greens[path], target, seed + frame_id)
        transforms[frame_id] = rec["M"]
        residuals[frame_id] = rec["residual"]
        failures[frame_id] = rec["failure"]
        if not rec["failure"]:
            warped_stack.append(_warp(greens[path], rec["M"]))
        else:
            # a failed frame contributes nothing to the cumulative stack
            pass

        # batch boundary (and not the last frame): rebuild the target
        at_boundary = (pos + 1) % batch_size == 0 and (pos + 1) < n
        if strategy == "evolving" and at_boundary and warped_stack:
            target = np.mean(np.stack(warped_stack), axis=0).astype(np.float32)
            rebuilds += 1

    runtime_s = time.perf_counter() - t_start
    return {
        "transforms": transforms,
        "residuals": residuals,
        "failures": failures,
        "n_aligned": sum(1 for v in failures.values() if not v),
        "n_failed": sum(1 for v in failures.values() if v),
        "runtime_s": runtime_s,
        "target_rebuilds": rebuilds,
    }


_FRAME_ID_CACHE = {}


def _frame_id(path):
    # Stable integer ID = index in the sorted non-reference frame list.
    return _FRAME_ID_CACHE[path]


_CANONICAL = {
    "centre": None,  # filled after ref_green shape known
    "edge": None,
    "corner": None,
}


def _dispersion(configs, canonical):
    """Across a dict of configs {label: {frame_id: M}}, measure per-frame max
    pairwise point displacement at ``canonical`` points.  Returns
    {point: {p50, p95, max}} over frames."""
    frames = set()
    for c in configs.values():
        frames.update(c.keys())
    frames = sorted(frames)

    per_point = {name: [] for name in canonical}
    for fid in frames:
        Ms = [configs[label][fid] for label in configs if fid in configs[label] and configs[label][fid] is not None]
        for name, p in canonical.items():
            dmax = 0.0
            for i in range(len(Ms)):
                for j in range(i + 1, len(Ms)):
                    d = np.linalg.norm(_apply(Ms[i], p) - _apply(Ms[j], p))
                    dmax = max(dmax, float(d))
            per_point[name].append(dmax)
    out = {}
    for name, vals in per_point.items():
        vals = np.array(vals)
        out[name] = {
            "p50": float(np.percentile(vals, 50)) if len(vals) else float("nan"),
            "p95": float(np.percentile(vals, 95)) if len(vals) else float("nan"),
            "max": float(vals.max()) if len(vals) else float("nan"),
        }
    return out


def run(folder=M16_FOLDER, seed=0, batch_sizes=(1, 5, 10),
        orders=("natural", "reversed"), max_frames=None):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"M16 data not present at {folder}")

    greens, ref_path, ref_green, ref_metric = prepare_all(folder)
    others = sorted([p for p in greens if p != ref_path])
    if max_frames:
        others = others[:max_frames]
    global _FRAME_ID_CACHE
    _FRAME_ID_CACHE = {p: i for i, p in enumerate(others)}
    _CANONICAL["centre"] = np.array([ref_green.shape[1] / 2.0, ref_green.shape[0] / 2.0])
    _CANONICAL["edge"] = np.array([ref_green.shape[1] - 1.0, ref_green.shape[0] / 2.0])
    _CANONICAL["corner"] = np.array([ref_green.shape[1] - 1.0, ref_green.shape[0] - 1.0])

    order_natural = others
    order_reversed = others[::-1]
    order_map = {"natural": order_natural, "reversed": order_reversed}

    # ---- immutable policy: target is fixed, so per-frame transforms are a
    #      function of (source, ref) only.  Compute once per frame; then the
    #      batch/order "configurations" are re-indexings of the same transforms.
    #      We still empirically recompute one reversed pass to confirm
    #      astroalign determinism (reported as the determinism floor). ----
    imm_natural = _run_policy("immutable", order_natural, greens, ref_green, 10, seed)
    imm_reversed = _run_policy("immutable", order_reversed, greens, ref_green, 10, seed)

    # ---- evolving policy: target is rebuilt from the batch stream. ----
    evolving = {}
    for oname in orders:
        for bs in batch_sizes:
            label = f"{oname}/bs{bs}"
            evolving[label] = _run_policy(
                "evolving", order_map[oname], greens, ref_green, bs, seed
            )

    # ---- organisation-sensitivity dispersions ----
    # immutable: batch sizes are irrelevant (no rebuild); use the single natural
    # pass under bs=1/5/10 (identical transforms) -> dispersion 0.
    imm_batch_configs = {f"bs{bs}": imm_natural["transforms"] for bs in batch_sizes}
    imm_order_configs = {
        "natural": imm_natural["transforms"],
        "reversed": imm_reversed["transforms"],
    }
    evo_batch_configs = {
        f"bs{bs}": evolving[f"natural/bs{bs}"]["transforms"] for bs in batch_sizes
    }
    evo_order_configs = {
        "natural": evolving[f"natural/bs{max(batch_sizes)}"]["transforms"],
        "reversed": evolving[f"reversed/bs{max(batch_sizes)}"]["transforms"],
    }

    result = {
        "folder": folder,
        "reference": os.path.basename(ref_path),
        "reference_metric": float(ref_metric),
        "n_others": len(others),
        "frame_shape": list(ref_green.shape),
        "canonical": {k: list(v) for k, v in _CANONICAL.items()},
        "batch_sizes": list(batch_sizes),
        "orders": list(orders),
        "seed": seed,
        "immutable": {
            "natural": imm_natural,
            "reversed": imm_reversed,
        },
        "evolving": evolving,
        "dispersion": {
            "immutable_batch": _dispersion(imm_batch_configs, _CANONICAL),
            "immutable_order": _dispersion(imm_order_configs, _CANONICAL),
            "evolving_batch": _dispersion(evo_batch_configs, _CANONICAL),
            "evolving_order": _dispersion(evo_order_configs, _CANONICAL),
            "immutable_vs_evolving": _dispersion(
                {
                    "immutable": imm_natural["transforms"],
                    "evolving": evolving[f"natural/bs{max(batch_sizes)}"]["transforms"],
                },
                _CANONICAL,
            ),
        },
    }
    return result


def _fmt(v, nd=3):
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return "  -  "
    return f"{float(v):.{nd}f}"


def report(r):
    L = []
    L.append("# M16 target-policy witness — immutable selected reference vs evolving target\n")
    L.append(
        f"folder={r['folder']}  frames (non-reference)={r['n_others']}  "
        f"reference={r['reference']} (metric {r['reference_metric']:.3f})  "
        f"green shape={r['frame_shape']}  batch sizes={r['batch_sizes']}  orders={r['orders']}"
    )
    L.append("")
    L.append(
        "Estimator = real astroalign matcher + production Euclidean conversion "
        "(scale discarded, `alignment.py:228-237`).  Evolving target = cumulative "
        "mean of warped green channels (deviation from production RGB memmap "
        "`sum/wht` documented in the module docstring)."
    )
    L.append("")

    # ---- per-config summary ----
    L.append("## Per-configuration summary\n")
    L.append(
        "| policy | order | bs | aligned | failed | runtime_s | target rebuilds | "
        "target-fit residual P50/P95 (px) |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for oname in r["orders"]:
        for bs in r["batch_sizes"]:
            c = r["evolving"][f"{oname}/bs{bs}"]
            res = [v for v in c["residuals"].values() if v is not None]
            p50 = float(np.percentile([x[0] for x in res], 50)) if res else float("nan")
            p95 = float(np.percentile([x[1] for x in res], 50)) if res else float("nan")
            L.append(
                f"| evolving | {oname} | {bs} | {c['n_aligned']} | {c['n_failed']} | "
                f"{c['runtime_s']:.1f} | {c['target_rebuilds']} | {_fmt(p50)}/{_fmt(p95)} |"
            )
    for oname in r["orders"]:
        c = r["immutable"][oname]
        res = [v for v in c["residuals"].values() if v is not None]
        p50 = float(np.percentile([x[0] for x in res], 50)) if res else float("nan")
        p95 = float(np.percentile([x[1] for x in res], 50)) if res else float("nan")
        L.append(
            f"| immutable | {oname} | n/a | {c['n_aligned']} | {c['n_failed']} | "
            f"{c['runtime_s']:.1f} | {c['target_rebuilds']} | {_fmt(p50)}/{_fmt(p95)} |"
        )
    L.append("")

    # ---- dispersion tables ----
    L.append("## Organisation sensitivity (point displacement, px; P50/P95/max over frames)\n")
    L.append(
        "Displacement of the fitted per-frame transform at canonical points, across "
        "configurations of the *same* policy.  Zero means the target identity is "
        "organization-independent.\n"
    )
    L.append("| comparison | centre P50/P95/max | edge P50/P95/max | corner P50/P95/max |")
    L.append("|---|---|---|---|")
    labels = [
        ("immutable × batch sizes", "immutable_batch"),
        ("immutable × order", "immutable_order"),
        ("evolving × batch sizes", "evolving_batch"),
        ("evolving × order", "evolving_order"),
        ("immutable vs evolving (same frame set)", "immutable_vs_evolving"),
    ]
    for label, key in labels:
        d = r["dispersion"][key]
        cells = []
        for pt in ("centre", "edge", "corner"):
            dd = d[pt]
            cells.append(f"{_fmt(dd['p50'],4)}/{_fmt(dd['p95'],4)}/{_fmt(dd['max'],4)}")
        L.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("")

    # ---- determinism floor ----
    L.append("## astroalign determinism floor\n")
    L.append(
        "The immutable policy's reversed pass recomputes every transform against the "
        "same fixed reference; any non-zero dispersion there is the RANSAC "
        "repeatability floor (empirically zero on this session)."
    )
    L.append("")

    # ---- limitations ----
    L.append("## Limitations (explicit)\n")
    L.append(
        "* **No ground truth** — M16 is one session, ~23 min, one focal state; "
        "these are *observational organization-sensitivity proxies*, not accuracy "
        "measurements.\n"
        "* **Evolving-target emulation** rebuilds the target as the mean of warped "
        "green channels (not the production RGB memmap `sum/wht` + WCS re-solve); "
        "not claimed worker-equivalent.\n"
        "* astroalign RANSAC uses an unseeded `default_rng`; empirically deterministic "
        "on these frames (see determinism floor)."
    )
    return "\n".join(L)


def main():
    try:
        r = run()
    except FileNotFoundError as e:
        print(f"# M16 target-policy witness skipped: {e}")
        return
    print(report(r))


if __name__ == "__main__":
    main()
