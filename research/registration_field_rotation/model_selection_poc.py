"""Deterministic synthetic model-selection POC for registration geometry.

RF-1 (corrective iteration C).  Production-external: this module does **not**
import ``seestar`` and does **not** touch the production pipeline.  The only
third-party imports are ``numpy``, ``skimage``, ``cv2`` (resampling witness)
and the external ``drizzle`` package (flux-redistribution witness) — none of
which triggers the ``seestar`` package's import side effects.

It compares the least-complex serious geometric registration models on
deterministic synthetic star fields with known ground-truth geometry, using a
strict fit-vs-held-out protocol.

Models (least -> most complex)
------------------------------
``translation``   2 dof   rigid translation
``euclidean``     3 dof   rotation + translation (scale forced to 1)
                         -> EXACTLY what production uses today (astroalign
                            SimilarityTransform with scale discarded)
``similarity``    4 dof   rotation + uniform scale + translation
                         -> what ``astroalign.find_transform`` actually returns
``affine``        6 dof   full affine (shear + anisotropic scale)
``projective``    8 dof   homography / perspective
``poly3``        20 dof   3rd-order polynomial (10 monomials x 2 coords)

``poly3`` is the *smooth* candidate.  A **degree-2** polynomial (skimage
``PolynomialTransform``, 12 dof) cannot represent the injected radial r^3
distortion: that field's Cartesian components contain cubic monomials
``x^3`` and ``x*y^2``, which are absent from a degree-2 basis.  A degree-3
polynomial (``[1, x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3]`` = 10 monomials
per coordinate = 20 dof total, minimum 10 points) spans those terms and is the
smallest smooth no-new-dependency candidate that can represent it.  Its
robustness cost is real: 20 free parameters demand a proper robust estimator
and are more fragile to false matches than the rigid/low-order models (see the
``outliers`` scenario).  Truly *local* (piecewise) models are deliberately
excluded for the same reason.

Determinism: every RNG is ``np.random.default_rng(seed)`` with a fixed seed per
scenario; fits are closed-form least squares / DLT (skimage
``estimate_transform``, or explicit normal-equation least squares for
``poly3``), with a single deterministic MAD rejection pass only for the
``outliers`` scenario.  Running the module twice yields bit-identical results.

The runtime reported by ``full_report`` is the measured min/median/max across
scenarios, not a fixed claim; microbenchmark timings are **non-decisive** and
kept only for completeness.
"""

from __future__ import annotations

import time

import numpy as np
from skimage.transform import estimate_transform

# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------

MODELS = ["translation", "euclidean", "similarity", "affine", "projective", "poly3"]

MODEL_INFO = {
    "translation": dict(dof=2, note="rigid translation"),
    "euclidean": dict(dof=3, note="rotation + translation (scale=1) — CURRENT PRODUCTION MODEL"),
    "similarity": dict(dof=4, note="rotation + uniform scale + translation — astroalign returns this"),
    "affine": dict(dof=6, note="full affine (shear + anisotropic scale)"),
    "projective": dict(dof=8, note="homography / perspective"),
    "poly3": dict(
        dof=20,
        note="3rd-order polynomial (10 monomials x 2 coords; smooth distortion proxy)",
    ),
}

MIN_POINTS = {
    "translation": 1,
    "euclidean": 2,
    "similarity": 2,
    "affine": 3,
    "projective": 4,
    "poly3": 10,
}

# Cubic (degree-3) monomial basis for the smooth polynomial candidate.
_CUBIC_MONOMIALS = ["1", "x", "y", "x2", "xy", "y2", "x3", "x2y", "xy2", "y3"]


def _cubic_basis(xy):
    """Return the (N, 10) degree-3 monomial design matrix."""
    xy = np.asarray(xy, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack(
        [
            np.ones_like(x),
            x,
            y,
            x * x,
            x * y,
            y * y,
            x ** 3,
            x * x * y,
            x * y * y,
            y ** 3,
        ]
    )


def apply_params(params, model, xy):
    """Apply a fitted model to ``(N, 2)`` ``(x, y)`` points.

    ``params`` conventions:
    * ``translation`` -> ``(tx, ty)``
    * euclidean/similarity/affine/projective -> 3x3 skimage matrix
    * ``poly3`` -> ``(2, 10)`` coefficient matrix (row 0 -> x', row 1 -> y')
    """
    xy = np.asarray(xy, dtype=np.float64)
    if model == "translation":
        return xy + np.asarray(params, dtype=np.float64)
    if model in ("euclidean", "similarity", "affine", "projective"):
        m = np.asarray(params, dtype=np.float64)
        hom = np.hstack([xy, np.ones((len(xy), 1))])
        out = hom @ m.T
        if model == "projective":
            return out[:, :2] / out[:, 2:3]
        return out[:, :2]
    if model == "poly3":
        c = np.asarray(params, dtype=np.float64)
        basis = _cubic_basis(xy)
        return np.column_stack([basis @ c[0], basis @ c[1]])
    raise ValueError(f"unknown model {model!r}")


def fit_model(model, src, dst):
    """Closed-form least-squares/DLT fit.  Returns params or ``None``."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < MIN_POINTS[model]:
        return None
    try:
        if model == "translation":
            return dst.mean(axis=0) - src.mean(axis=0)
        if model in ("euclidean", "similarity", "affine", "projective"):
            return estimate_transform(model, src, dst).params
        if model == "poly3":
            # Explicit least squares: solve A @ C = dst for C (10 x 2).
            A = _cubic_basis(src)
            coef, *_ = np.linalg.lstsq(A, dst, rcond=None)
            return coef.T  # (2, 10)
    except Exception:
        return None
    return None


def residuals(params, model, src, dst):
    """Euclidean residual norm per point."""
    return np.linalg.norm(apply_params(params, model, src) - np.asarray(dst), axis=1)


def fit_with_rejection(model, src, dst, k=5.0, max_iter=4):
    """Deterministic MAD-based outlier rejection around a closed-form fit.

    Returns ``(params, inlier_mask)``.  The mask only ever shrinks.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    sel = np.ones(len(src), dtype=bool)
    params = None
    for _ in range(max_iter):
        if sel.sum() < MIN_POINTS[model]:
            return None, sel
        params = fit_model(model, src[sel], dst[sel])
        if params is None:
            return None, sel
        r = residuals(params, model, src, dst)
        med = np.median(r[sel])
        mad = np.median(np.abs(r[sel] - med))
        sigma = 1.4826 * mad if mad > 0 else 1e-9
        new_sel = r <= med + k * sigma
        # never re-admit points; stop when stable
        new_sel = new_sel & sel
        if new_sel.sum() == sel.sum():
            break
        sel = new_sel
    return params, sel


# --------------------------------------------------------------------------
# Ground-truth geometry generators  (rng, src) -> dst
# --------------------------------------------------------------------------


def _rot(theta_deg):
    t = np.radians(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def gt_translation(dx, dy):
    def f(rng, src):
        return src + np.array([dx, dy])
    return f


def gt_rotation(theta_deg, centre):
    def f(rng, src):
        c = np.asarray(centre, dtype=np.float64)
        return (src - c) @ _rot(theta_deg).T + c
    return f


def gt_similarity(scale, theta_deg, centre, t=(0.0, 0.0)):
    def f(rng, src):
        c = np.asarray(centre, dtype=np.float64)
        m = scale * _rot(theta_deg)
        return (src - c) @ m.T + c + np.asarray(t)
    return f


def gt_affine(A, t):
    A = np.asarray(A, dtype=np.float64)

    def f(rng, src):
        return src @ A.T + np.asarray(t)
    return f


def gt_projective(H):
    H = np.asarray(H, dtype=np.float64)

    def f(rng, src):
        hom = np.hstack([src, np.ones((len(src), 1))]) @ H.T
        return hom[:, :2] / hom[:, 2:3]
    return f


def gt_radial(k, centre, rmax):
    """Smooth radial distortion: ``r' = r * (1 + k * (r/rmax)^2)``.

    The displacement ``r * k * (r/rmax)^2 = k * r^3 / rmax^2`` is **cubic** in
    radius; in detector coordinates its components contain the monomials
    ``x^3`` and ``x*y^2`` (resp. ``y^3``, ``x^2 y``), so only a degree-3 (or
    higher) polynomial can represent it.  A degree-2 polynomial cannot.
    """
    def f(rng, src):
        c = np.asarray(centre, dtype=np.float64)
        d = src - c
        r = np.linalg.norm(d, axis=1)
        rn = r / rmax
        factor = 1.0 + k * (rn ** 2)
        return c + d * factor[:, None]
    return f


# --------------------------------------------------------------------------
# Scenario definitions
# --------------------------------------------------------------------------

# Seestar S50 native sensor size (approx).
FIELD_W, FIELD_H = 1920.0, 1080.0
CENTRE = (FIELD_W / 2.0, FIELD_H / 2.0)
RMAX = float(np.hypot(FIELD_W / 2.0, FIELD_H / 2.0))  # ~1101.5 px


def _build_scenarios():
    """Return an ordered dict name -> scenario config (all deterministic)."""
    s = {}
    s["translation"] = dict(
        n_stars=100, sigma=0.05, true=gt_translation(12.3, -8.7), seed=1
    )
    s["rotation"] = dict(
        n_stars=100, sigma=0.05, true=gt_rotation(30.0, CENTRE), seed=2
    )
    s["rotation_translation"] = dict(
        n_stars=100, sigma=0.05,
        true=_compose(gt_rotation(18.0, CENTRE), gt_translation(25.0, -15.0)),
        seed=3,
    )
    s["large_rotation"] = dict(
        n_stars=100, sigma=0.05, true=gt_rotation(120.0, CENTRE), seed=4
    )
    s["scale"] = dict(
        n_stars=100, sigma=0.05,
        true=gt_similarity(1.003, 1.0, CENTRE, t=(10.0, -5.0)), seed=5,
        note="0.3% uniform field-scale drift (thermal focus proxy)",
    )
    s["affine"] = dict(
        n_stars=100, sigma=0.05,
        true=gt_affine([[1.004, 0.003], [-0.002, 1.001]], (8.0, -6.0)), seed=6,
        note="small shear + anisotropic scale (differential refraction proxy)",
    )
    s["projective"] = dict(
        n_stars=100, sigma=0.05,
        true=gt_projective(
            [[1.0002, 0.0001, 3.0], [-0.0001, 1.0001, -2.0],
             [1.5e-6, -1.0e-6, 1.0]]
        ),
        seed=7,
        note="mild keystone/perspective",
    )
    s["smooth_local"] = dict(
        n_stars=100, sigma=0.05,
        true=_compose(gt_rotation(5.0, CENTRE), gt_radial(0.002, CENTRE, RMAX)),
        seed=8,
        note="smooth radial (r^3) optical distortion, ~2.2 px at corners",
    )
    s["partial_overlap"] = dict(
        n_stars=100, sigma=0.05,
        true=gt_rotation(15.0, CENTRE), seed=9,
        keep=dict(xmax=0.55 * FIELD_W),
        note="matches restricted to the left 55% of the field",
    )
    s["outliers"] = dict(
        n_stars=100, sigma=0.05,
        true=_compose(gt_rotation(18.0, CENTRE), gt_translation(25.0, -15.0)),
        seed=10, outlier_frac=0.15, reject=True,
        note="15% false matches (random correspondences)",
    )
    s["degenerate"] = dict(
        n_stars=3, sigma=0.05, true=gt_translation(5.0, 5.0), seed=11,
        note="only 3 stars (2 in fit set) — underdetermined for affine/projective/poly3",
    )
    return s


def _compose(g, f):
    def h(rng, src):
        return g(rng, f(rng, src))
    return h


SCENARIOS = _build_scenarios()


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _regions(xy, centre, rmax):
    """Classify points into centre / edge / corner by radius fraction."""
    r = np.linalg.norm(np.asarray(xy) - np.asarray(centre), axis=1) / rmax
    centre_mask = r < 0.33
    corner_mask = r >= 0.67
    edge_mask = ~centre_mask & ~corner_mask
    return centre_mask, edge_mask, corner_mask


def run_scenario(name, cfg):
    """Fit every model on one scenario and return a metrics dict."""
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["n_stars"]
    src = rng.uniform([0.0, 0.0], [FIELD_W, FIELD_H], size=(n, 2))
    dst_true = cfg["true"](rng, src)
    dst = dst_true + rng.normal(0.0, cfg["sigma"], size=(n, 2))

    # deterministic shuffle split -> fit / held-out
    order = rng.permutation(n)
    n_fit = max(1, int(round(n * 0.7)))
    fit_idx = order[:n_fit]
    hold_idx = order[n_fit:]

    # partial overlap: keep only stars whose *source* falls in the keep region
    if cfg.get("keep"):
        keep_mask = src[:, 0] <= cfg["keep"]["xmax"]
        fit_idx = fit_idx[keep_mask[fit_idx]]
        hold_idx = hold_idx[keep_mask[hold_idx]]

    # outliers: corrupt a fraction of the *fit* set with false matches
    n_out = 0
    if cfg.get("outlier_frac"):
        n_out = max(1, int(round(len(fit_idx) * cfg["outlier_frac"])))
        out_sel = rng.choice(fit_idx, size=n_out, replace=False)
        dst[out_sel] = rng.uniform([0.0, 0.0], [FIELD_W, FIELD_H], size=(n_out, 2))
        true_inlier_fit = np.ones(len(fit_idx), dtype=bool)
        true_inlier_fit[np.isin(fit_idx, out_sel)] = False

    src_fit, dst_fit = src[fit_idx], dst[fit_idx]
    src_hold, dst_hold = src[hold_idx], dst[hold_idx]

    results = {}
    for model in MODELS:
        r = {
            "model": model,
            "fit_ok": False,
            "n_fit": len(src_fit),
            "n_inliers": None,
            "fit_rms": np.nan,
            "fit_p95": np.nan,
            "hold_rms": np.nan,
            "hold_p50": np.nan,
            "hold_p95": np.nan,
            "centre": np.nan,
            "edge": np.nan,
            "corner": np.nan,
            "time_us": np.nan,
        }
        t0 = time.perf_counter()
        if cfg.get("reject"):
            params, inl = fit_with_rejection(model, src_fit, dst_fit)
        else:
            params = fit_model(model, src_fit, dst_fit)
            inl = None
        dt = time.perf_counter() - t0

        if params is None:
            r["time_us"] = dt * 1e6
            results[model] = r
            continue

        r["fit_ok"] = True
        r["time_us"] = dt * 1e6

        rfit = residuals(params, model, src_fit, dst_fit)
        r["fit_rms"] = float(np.sqrt(np.mean(rfit ** 2)))
        r["fit_p95"] = float(np.percentile(rfit, 95))
        if inl is not None:
            r["n_inliers"] = int(inl.sum())

        if len(src_hold):
            rh = residuals(params, model, src_hold, dst_hold)
            r["hold_rms"] = float(np.sqrt(np.mean(rh ** 2)))
            r["hold_p50"] = float(np.percentile(rh, 50))
            r["hold_p95"] = float(np.percentile(rh, 95))
            cm, em, km = _regions(src_hold, CENTRE, RMAX)
            r["centre"] = float(np.mean(rh[cm])) if cm.any() else np.nan
            r["edge"] = float(np.mean(rh[em])) if em.any() else np.nan
            r["corner"] = float(np.mean(rh[km])) if km.any() else np.nan

        results[model] = r

    return {
        "name": name,
        "note": cfg.get("note", ""),
        "n_fit": len(src_fit),
        "n_hold": len(src_hold),
        "n_outliers": n_out,
        "results": results,
    }


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  -  "
    return f"{v:>{nd}.{max(nd-1,1)}f}" if abs(v) < 1000 else f"{v:>{nd}.1e}"


def run_all():
    """Run every scenario, return list of per-scenario dicts."""
    return [run_scenario(name, cfg) for name, cfg in SCENARIOS.items()]


def summary_table(runs):
    """Return a markdown string summarising held-out RMS (px) per model/scenario."""
    header = "| scenario | " + " | ".join(MODELS) + " |"
    sep = "|---|" + "|".join(["---"] * len(MODELS)) + "|"
    rows = [header, sep]
    for run in runs:
        cells = []
        for m in MODELS:
            r = run["results"][m]
            if not r["fit_ok"]:
                cells.append("FAIL")
            else:
                cells.append(_fmt(r["hold_rms"]).strip())
        rows.append(f"| {run['name']} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def runtime_summary(runs):
    """Honest min/median/max fit wall-time (us) per model across scenarios.

    Microbenchmark timings are non-decisive; this exists only to correct the
    earlier over-precise claim.  Scenarios that failed to fit are skipped.
    """
    lines = ["| model | min us | median us | max us | n (fitted scenarios) |",
             "|---|---|---|---|---|"]
    for m in MODELS:
        ts = []
        for run in runs:
            r = run["results"][m]
            if r["fit_ok"] and not np.isnan(r["time_us"]):
                ts.append(r["time_us"])
        if not ts:
            lines.append(f"| {m} | - | - | - | 0 |")
        else:
            lines.append(
                f"| {m} | {min(ts):.1f} | {np.median(ts):.1f} | {max(ts):.1f} | {len(ts)} |"
            )
    return "\n".join(lines)


def full_report():
    """Return a multi-line text report (markdown)."""
    runs = run_all()
    lines = []
    lines.append("# RF-1 synthetic model-selection POC — measured report (corrective C)\n")
    lines.append(f"field={FIELD_W}x{FIELD_H} px, centre={CENTRE}, corner radius={RMAX:.1f} px")
    lines.append(f"noise sigma={0.05:.3f} px per centroid, 70/30 fit/hold-out split, seeds fixed\n")
    lines.append("## Held-out RMS residual (px) per model per scenario\n")
    lines.append(summary_table(runs))
    lines.append("\n## Full metrics\n")
    for run in runs:
        lines.append(f"### {run['name']}  ({run['note']})")
        lines.append(
            f"fit={run['n_fit']}, held-out={run['n_hold']}, injected outliers={run['n_outliers']}"
        )
        lines.append(
            "| model | fit_rms | fit_p95 | hold_rms | hold_p50 | hold_p95 | centre | edge | corner | inliers | us |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for m in MODELS:
            r = run["results"][m]
            lines.append(
                f"| {m} | {_fmt(r['fit_rms'])} | {_fmt(r['fit_p95'])} | "
                f"{_fmt(r['hold_rms'])} | {_fmt(r['hold_p50'])} | {_fmt(r['hold_p95'])} | "
                f"{_fmt(r['centre'])} | {_fmt(r['edge'])} | {_fmt(r['corner'])} | "
                f"{r['n_inliers'] if r['n_inliers'] is not None else '-'} | {_fmt(r['time_us'],2)} |"
            )
        lines.append("")
    lines.append("## Fit wall-time per model (min/median/max us; non-decisive)\n")
    lines.append(runtime_summary(runs))
    lines.append("\n## Radial residual structure (smooth_local scenario)\n")
    lines.append(_radial_structure_report())
    lines.append("\n## Resampling / deposition effect (single synthetic star)\n")
    lines.append(_resampling_report())
    return "\n".join(lines)


def _radial_structure_report():
    """Correlation between residual magnitude and radius for each model on the
    smooth radial-distortion scenario.  A high value supports a stable
    detector-coordinate (radial) distortion hypothesis being left unmodeled."""
    cfg = SCENARIOS["smooth_local"]
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["n_stars"]
    src = rng.uniform([0.0, 0.0], [FIELD_W, FIELD_H], size=(n, 2))
    dst_true = cfg["true"](rng, src)
    dst = dst_true + rng.normal(0.0, cfg["sigma"], size=(n, 2))
    order = rng.permutation(n)
    n_fit = max(1, int(round(n * 0.7)))
    hold_idx = order[n_fit:]
    src_hold, dst_hold = src[hold_idx], dst[hold_idx]
    r = np.linalg.norm(src_hold - np.asarray(CENTRE), axis=1)

    lines = []
    lines.append(
        "| model | Spearman(r, resid) | slope px/1000px | hold_rms px |"
    )
    lines.append("|---|---|---|---|")
    for m in MODELS:
        params = fit_model(m, src[order[:n_fit]], dst[order[:n_fit]])
        if params is None:
            lines.append(f"| {m} | FAIL | - | - |")
            continue
        rh = residuals(params, m, src_hold, dst_hold)
        # rank correlation between radius and residual magnitude
        rs = np.corrcoef(np.argsort(np.argsort(r)), np.argsort(np.argsort(rh)))[0, 1]
        # linear slope of residual vs radius (px per 1000 px of radius)
        slope = np.polyfit(r, rh, 1)[0] * 1000.0
        lines.append(
            f"| {m} | {rs:+.3f} | {slope:+.2f} | {_fmt(np.sqrt(np.mean(rh**2)))} |"
        )
    return "\n".join(lines)


def recovered_similarity_scale():
    """Diagnostic: the uniform scale that a similarity (astroalign-like) fit
    recovers in the ``scale`` scenario, versus the ``1.0`` that the production
    code forces when it discards astroalign's scale."""
    cfg = SCENARIOS["scale"]
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["n_stars"]
    src = rng.uniform([0.0, 0.0], [FIELD_W, FIELD_H], size=(n, 2))
    dst_true = cfg["true"](rng, src)
    dst = dst_true + rng.normal(0.0, cfg["sigma"], size=(n, 2))
    order = rng.permutation(n)
    n_fit = max(1, int(round(n * 0.7)))
    m = estimate_transform("similarity", src[order[:n_fit]], dst[order[:n_fit]])
    scale = float(np.hypot(m.params[0, 0], m.params[1, 0]))
    return scale


def _moments(img, xx, yy):
    img = np.asarray(img, dtype=np.float64)
    tot = img.sum()
    if tot <= 0:
        return tot, 32.0, 32.0, 0.0, 0.0, 0.0
    cx = (img * xx).sum() / tot
    cy = (img * yy).sum() / tot
    sxx = (img * (xx - cx) ** 2).sum() / tot
    syy = (img * (yy - cy) ** 2).sum() / tot
    sxy = (img * (xx - cx) * (yy - cy)).sum() / tot
    return tot, cx, cy, sxx, syy, sxy


def _fwhm_ecc(img, xx, yy):
    tot, cx, cy, sxx, syy, sxy = _moments(img, xx, yy)
    cov = np.array([[sxx, sxy], [sxy, syy]])
    ev = np.linalg.eigvalsh(cov)
    a, b = np.sqrt(np.sort(ev)[::-1])
    fwhm = 2.355 * np.sqrt((a ** 2 + b ** 2) / 2.0)
    ecc = 0.0 if a < 1e-12 else float(np.sqrt(1.0 - (b / a) ** 2))
    return tot, cx, cy, fwhm, ecc, float(img.max())


def _resampling_report():
    """Measured PSF effects of the production warp path (cv2.warpAffine
    INTER_LINEAR) vs the drizzle flux-redistribution path, for a single
    synthetic star.  This distinguishes *data interpolation* (warpAffine) from
    *flux redistribution / deposition* (drizzle): drizzle does not
    pre-interpolate the source data, but its flux redistribution into the
    output grid is itself the one sampling/deposition stage for that path.

    The drizzle row is measured with the **external** ``drizzle`` package
    (``drizzle.resample.Drizzle``) — exactly the kernel wrapped by the
    production ``DrizzleAccumulator`` — so this module still does not import
    ``seestar``.  (Kernel/PSF study is deferred to the RF-2 implementation
    gate; this is a compact sanity measurement.)
    """
    import cv2

    from drizzle.resample import Drizzle

    shape = (64, 64)
    yy, xx = np.indices(shape)
    sig = 1.5  # sharper star so interpolation blur is visible
    star = 1000.0 * np.exp(-((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / (2.0 * sig ** 2))

    lines = []
    lines.append("star sigma=1.5px, peak=1000, no noise. metrics = flux/peak/FWHM/ecc/centroid")
    lines.append("| method | flux | peak | FWHM(px) | ecc | centroid |")
    lines.append("|---|---|---|---|---|---|")

    def _row(desc, img):
        tot, cx, cy, fwhm, ecc, peak = _fwhm_ecc(img, xx, yy)
        lines.append(
            f"| {desc} | {tot:.2f} | {peak:.2f} | {fwhm:.3f} | {ecc:.3f} | ({cx:.2f},{cy:.2f}) |"
        )

    # identity
    _row("identity (no resampling)", star.copy())
    # diagonal 0.3 px sub-pixel shift (both axes symmetric -> no phase artifact)
    m_shift = np.array([[1, 0, 0.3], [0, 1, 0.3]], dtype=np.float32)
    _row(
        "warpAffine INTER_LINEAR (production CPU data interpolation)",
        cv2.warpAffine(star, m_shift, (64, 64), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0),
    )
    _row(
        "warpAffine INTER_CUBIC",
        cv2.warpAffine(star, m_shift, (64, 64), flags=cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0),
    )
    _row(
        "warpAffine INTER_LANCZOS4",
        cv2.warpAffine(star, m_shift, (64, 64), flags=cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0),
    )
    # drizzle pixmap deposition (square kernel, pixfrac 1.0) via external drizzle
    pixmap = np.dstack((xx + 0.3, yy + 0.3)).astype(np.float64)
    in_grid = (pixmap[..., 0] < 64) & (pixmap[..., 0] >= 0) & \
              (pixmap[..., 1] < 64) & (pixmap[..., 1] >= 0)
    out_img = np.zeros(shape, np.float32)
    out_wht = np.zeros(shape, np.float32)
    dz = Drizzle(out_img=out_img, out_wht=out_wht, kernel="square", fillval="0.0")
    dz.add_image(
        data=star.astype(np.float32),
        exptime=1.0,
        pixmap=pixmap,
        weight_map=np.ones(shape, np.float32) * in_grid.astype(np.float32),
        in_units="counts",
        pixfrac=1.0,
        wht_scale=1.0,
    )
    sci = (out_img * out_wht).astype(np.float32)
    drizzle_out = np.nan_to_num(sci / np.maximum(out_wht, 1e-9),
                                nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    _row("drizzle pixmap deposition (square, no data interpolation)", drizzle_out)
    return "\n".join(lines)


def main():
    print(full_report())


if __name__ == "__main__":
    main()
