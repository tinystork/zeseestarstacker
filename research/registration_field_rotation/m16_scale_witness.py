"""M16 real-data witness: is the discarded astroalign similarity scale real?

RF-1 (corrective iteration C).  Production-external: this script does **not**
import ``seestar``; it re-implements the production preparation/detection/
matching *basis* with the same libraries the production code uses (``astropy``
+ ``cv2`` debayer/hot-pixel + ``astroalign.find_transform``) so that no
production import side effects occur.  It is a *complementary* witness, never
the primary evidence for the model decision.

Production preparation basis faithfully replicated (read-only reimplementation,
no ``seestar`` import)
-------------------------------------------------------------------------------
The production per-frame preparation (``queue_manager._process_file``) and
reference selection (``alignment.SeestarAligner._get_reference_image``) are
re-implemented here, with the two defects of the previous iteration corrected:

1. **BAYERPAT from the FITS header.**  The previous iteration hardcoded
   ``GRBG``.  Production reads ``header['BAYERPAT']`` (falling back to the
   configured ``bayer_pattern``); here we read ``BAYERPAT`` per frame (M16
   frames all carry ``BAYERPAT='GRBG'``, so the numbers do not change, but the
   method is now faithful).
2. **Hot-pixel correction.**  Production applies
   ``detect_and_correct_hot_pixels(img, threshold=3.0, neighborhood_size=5)``
   to the debayered RGB image when ``correct_hot_pixels`` is enabled (it is
   enabled by default: ``SeestarAligner.correct_hot_pixels = True``).  The
   previous iteration omitted this step.  Here we replicate the *deterministic
   CPU* path of ``detect_and_correct_hot_pixels`` bit-faithfully
   (``cv2.medianBlur`` median + ``cv2.blur`` box-filter mean/mean_sq +
   ``hot = channel > median + threshold*std`` + replace by median).  The only
   production branch not replicated is the CUDA box filter (unavailable here);
   the CUDA vs CPU box filter differs only at float rounding, far below the
   scale conclusion's noise.  This is stated as a deviation and verified by the
   A/B in §7 of the research report.

Note on white balance: production ``_process_file`` applies a basic green-gain
white balance to the R and B channels **after** debayer.  The **green** channel
— the only channel used for ``find_transform`` here and in production — is
**not** modified by that step, and the production reference-selection metric is
computed on the debayered RGB **without** white balance.  So white balance is
green-invariant for this witness and is correctly omitted.

What it does
------------
1. Loads the 20 read-only FITS frames in ``/home/tristan/M16/quick`` (one
   session, 2025-05-30, ~23 min).
2. Replicates the production reference selection: for each candidate, normalize
   to [0,1], reject by low variance (``std < 0.0005``), debayer with header
   ``BAYERPAT``, hot-pixel-correct, then score by the production quality metric
   ``median / (1.4826 * MAD)`` (same formula as ``_get_reference_image``); the
   highest-scoring frame is the reference.
3. For every other frame, runs the production detection/matching basis:
   normalize to [0,1], debayer with header ``BAYERPAT``, hot-pixel-correct,
   green channel, then ``astroalign.find_transform(source=green,
   target=reference_green)`` which returns the matched star pairs
   ``(source_matches, target_matches)`` (no correspondences fabricated).
4. Splits the astroalign matched pairs deterministically (seeded, 70/30
   fit/hold-out) and fits **Euclidean** (scale=1 — the current production
   model) vs **Similarity** on the fit subset, evaluating each on the
   held-out subset only (never the points used for the fit).
5. Reports per-frame returned scale, fit/held-out P50/P95/RMS, centre/edge/
   corner residuals, rotation, translation, match counts/failures, and
   runtime; then aggregates the scale range/median/MAD (using the **correct**
   statistics — see below) and the held-out improvement attributable to scale.

Determinism: a single master seed (``--seed``) drives every per-frame split;
no RNG is unseeded.  Re-running yields bit-identical results.

Hold-out limitation (stated explicitly)
---------------------------------------
astroalign's matched-pair selection / RANSAC is run on **all** detected stars
*before* the 70/30 fit/hold-out split.  The held-out residuals are therefore a
**model-fit hold-out** (the fit never sees the held-out pairs), *not* a fully
independent correspondence-selection validation: the correspondences themselves
were chosen by astroalign using every star.  This witness measures whether a
similarity scale is *supported by the matched pairs astroalign actually
produced*; it does not independently validate the matcher.

Statistics correction (defect #1)
---------------------------------
The previous label "median |scale-1|" was wrong: the code computed
``abs(median(scale) - 1)``.  This iteration reports all three, correctly
labelled:

* ``|median(scale) - 1|`` — the offset of the median returned scale from 1.
* ``median(|scale - 1|)`` — the median absolute per-frame scale deviation.
* ``mean(|scale - 1|)`` — the mean absolute per-frame scale deviation.

The corner-pixel implication is stated using ``median(|scale - 1|)`` (the
correct "typical per-frame" deviation), not ``|median(scale)-1|``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

# --------------------------------------------------------------------------
# Production preparation basis (mirrors seestar.core.image_processing and
# seestar.core.hot_pixels without importing seestar).
# --------------------------------------------------------------------------


def _load_normalized(path):
    """Mirror of ``load_and_validate_fits`` (normalize_to_float32=True): return
    the first image HDU normalized to float32 [0,1], or ``None`` on failure."""
    from astropy.io import fits

    with fits.open(path, memmap=False, do_not_scale_image_data=True) as hdul:
        hdu = None
        for idx, h in enumerate(hdul):
            if h.is_image and hasattr(h, "data") and h.data is not None:
                if idx == 0 or (hasattr(h, "name") and isinstance(h.name, str)
                                and h.name.upper() in ("SCI", "IMAGE", "PRIMARY")):
                    hdu = h
                    break
        if hdu is None:
            for h in hdul:
                if h.is_image and hasattr(h, "data") and h.data is not None:
                    hdu = h
                    break
        if hdu is None:
            return None, None
        data = hdu.data
        header = hdu.header.copy()
        if data.ndim == 3:
            if data.shape[0] in (1, 3, 4) and data.shape[1] > 4 and data.shape[2] > 4:
                data = np.moveaxis(data, 0, -1)
            elif not (data.shape[2] in (1, 3, 4) and data.shape[0] > 4
                      and data.shape[1] > 4):
                return None, header
        elif data.ndim != 2:
            return None, header
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        d = data.astype(np.float64)
        mn, mx = np.nanmin(d), np.nanmax(d)
        if not (np.isfinite(mn) and np.isfinite(mx) and mx > mn):
            if np.any(np.isfinite(d)):
                norm = np.full_like(data, 0.5, dtype=np.float32)
            else:
                norm = np.zeros_like(data, dtype=np.float32)
            return norm, header
        norm = (data.astype(np.float32) - np.float32(mn)) / np.float32(mx - mn)
        return np.clip(norm, 0.0, 1.0).astype(np.float32), header


def _debayer(img, pattern):
    """Mirror of ``debayer_image`` (float32 0-1 -> float32 0-1 RGB)."""
    import cv2

    codes = {
        "GRBG": cv2.COLOR_BayerGR2RGB,
        "RGGB": cv2.COLOR_BayerRG2RGB,
        "GBRG": cv2.COLOR_BayerGB2RGB,
        "BGGR": cv2.COLOR_BayerBG2RGB,
    }
    u = (np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)
    bgr = cv2.cvtColor(u, codes[pattern.upper()])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0


def _detect_and_correct_hot_pixels_cpu(image, threshold=3.0, neighborhood_size=5):
    """Faithful CPU-path reimplementation of ``detect_and_correct_hot_pixels``.

    Production branch order preserved: medianBlur (median reference), blur
    box-filter mean and mean_sq, std = sqrt(max(mean_sq - mean^2, 1e-10)),
    std floor 1e-5 for float images, hot = channel > median + threshold*std,
    replace by median.  Color images are processed channel-by-channel.  The
    production CUDA branch (``cv2.cuda`` box filter) is not replicated; it
    differs only at float rounding.
    """
    import cv2

    if neighborhood_size % 2 == 0:
        neighborhood_size += 1
    neighborhood_size = max(3, neighborhood_size)
    ksize = (neighborhood_size, neighborhood_size)

    original_dtype = image.dtype
    img = image.astype(np.float32, copy=True)
    is_color = img.ndim == 3 and img.shape[-1] == 3

    def _correct(channel):
        med = cv2.medianBlur(channel, neighborhood_size)
        mean = cv2.blur(channel, ksize)
        mean_sq = cv2.blur(channel ** 2, ksize)
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 1e-10))
        std_floor = 1.0 if np.issubdtype(original_dtype, np.integer) else 1e-5
        std = np.maximum(std, std_floor)
        mask = channel > (med + threshold * std)
        channel[mask] = med[mask]

    if is_color:
        for c in range(img.shape[2]):
            _correct(img[:, :, c])
    else:
        _correct(img)

    if np.issubdtype(original_dtype, np.integer):
        mn, mx = np.iinfo(original_dtype).min, np.iinfo(original_dtype).max
        return np.clip(img, mn, mx).astype(original_dtype)
    return img.astype(original_dtype)


def _quality_metric(rgb):
    """Production reference-selection metric: median / (1.4826 * MAD)."""
    med = float(np.median(rgb))
    mad = float(np.median(np.abs(rgb - med)))
    return med / (1.4826 * mad + 1e-9) if med > 1e-9 else -np.inf


def _green(rgb):
    return rgb[..., 1]


def _bayer_pattern_from_header(header, default="GRBG"):
    pat = header.get("BAYERPAT", default) if header is not None else default
    return pat.upper() if isinstance(pat, str) and pat.upper() in (
        "GRBG", "RGGB", "GBRG", "BGGR") else default.upper()


# --------------------------------------------------------------------------
# Model fitting (skimage, no new dependency)
# --------------------------------------------------------------------------


def _fit(model, src, dst):
    from skimage.transform import estimate_transform

    return estimate_transform(model, src, dst)


def _resid(model, src, dst):
    pred = model(src)
    return np.linalg.norm(pred - dst, axis=1)


def _regions(xy, centre, rmax):
    r = np.linalg.norm(np.asarray(xy) - np.asarray(centre), axis=1) / rmax
    return r < 0.33, (r >= 0.33) & (r < 0.67), r >= 0.67


def scale_statistics(scales, rmax):
    """Correctly-labelled scale-deviation statistics.

    Returns ``(stats_ppm, corner_px)`` where ``stats_ppm`` is a dict with
    * ``abs_median_minus_1`` : ``|median(scale) - 1|`` in ppm,
    * ``median_abs``        : ``median(|scale - 1|)`` in ppm,
    * ``mean_abs``          : ``mean(|scale - 1|)`` in ppm,
    and ``corner_px`` maps the same keys to their corner displacement
    ``ppm * 1e-6 * rmax`` in px.  The three quantities are **different** and
    must not be conflated (the previous "median |scale-1|" label was wrong).
    """
    scales = np.asarray(scales, dtype=np.float64)
    med = float(np.median(scales))
    dev = np.abs(scales - 1.0)
    stats_ppm = {
        "abs_median_minus_1": abs(med - 1.0) * 1e6,
        "median_abs": float(np.median(dev)) * 1e6,
        "mean_abs": float(np.mean(dev)) * 1e6,
    }
    corner_px = {k: v * 1e-6 * float(rmax) for k, v in stats_ppm.items()}
    return stats_ppm, corner_px


# --------------------------------------------------------------------------
# Preparation of a single frame (reference selection and per-frame alignment
# share this; WB is green-invariant and omitted).
# --------------------------------------------------------------------------


def _prepare(path):
    norm, header = _load_normalized(path)
    if norm is None:
        return None, header, None
    if norm.ndim != 2:
        return norm, header, None
    pattern = _bayer_pattern_from_header(header)
    rgb = _debayer(norm, pattern)
    rgb = _detect_and_correct_hot_pixels_cpu(rgb, threshold=3.0, neighborhood_size=5)
    return rgb, header, pattern


def _variance_ok(norm):
    return float(np.std(norm)) >= 0.0005


# --------------------------------------------------------------------------
# Main witness
# --------------------------------------------------------------------------


def run(folder, seed=0, fit_frac=0.7, hot_pixels=True, header_bayer=True):
    import astroalign as aa

    files = sorted(glob.glob(os.path.join(folder, "*.fit")))
    assert files, f"no .fit frames in {folder}"

    # 1. Prepare all frames once (deterministic order = sorted filename),
    #    tracking the debayered pre-hot-pixel image too (for the A/B).
    prepared = {}
    patterns = {}
    for f in files:
        norm, header = _load_normalized(f)
        if norm is None:
            prepared[f] = None
            continue
        if norm.ndim != 2:
            prepared[f] = None
            continue
        pattern = _bayer_pattern_from_header(header) if header_bayer else "GRBG"
        patterns[f] = pattern
        rgb = _debayer(norm, pattern)
        if hot_pixels:
            rgb = _detect_and_correct_hot_pixels_cpu(rgb, 3.0, 5)
        prepared[f] = rgb

    # 2. Reference selection: production quality metric argmax over candidates
    #    that pass the variance gate.
    candidates = []
    for f in files:
        rgb = prepared.get(f)
        if rgb is None:
            continue
        norm, _ = _load_normalized(f)
        if not _variance_ok(norm):
            continue
        candidates.append((f, _quality_metric(rgb)))
    assert candidates, "no reference candidate passed"
    ref_path, ref_metric = max(candidates, key=lambda t: t[1])
    ref_rgb = prepared[ref_path]
    ref_green = _green(ref_rgb)
    others = [f for f in files if f != ref_path]
    centre = (ref_green.shape[1] / 2.0, ref_green.shape[0] / 2.0)
    rmax = float(np.hypot(centre[0], centre[1]))

    # 3. Per-frame alignment + held-out Euclidean vs Similarity.
    rows = []
    failures = []
    for i, f in enumerate(others):
        rgb = prepared[f]
        rec = {
            "file": os.path.basename(f),
            "bayerpat": patterns.get(f),
            "match_count": None,
            "astroalign_scale": None,
            "rotation_deg": None,
            "tx": None,
            "ty": None,
            "runtime_s": None,
            "fit_n": None,
            "hold_n": None,
            "euclidean": {},
            "similarity": {},
            "status": "ok",
        }
        if rgb is None:
            rec["status"] = "load_failed"
            failures.append(rec)
            rows.append(rec)
            continue

        src_green = _green(rgb)
        t0 = time.perf_counter()
        try:
            T, (s, t) = aa.find_transform(source=src_green, target=ref_green)
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"match_failed:{type(e).__name__}"
            rec["runtime_s"] = time.perf_counter() - t0
            failures.append(rec)
            rows.append(rec)
            continue
        rec["runtime_s"] = time.perf_counter() - t0

        s = np.asarray(s, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        rec["match_count"] = int(len(s))
        rec["astroalign_scale"] = float(np.hypot(T.params[0, 0], T.params[1, 0]))
        rec["rotation_deg"] = float(np.degrees(np.arctan2(T.params[1, 0], T.params[0, 0])))
        rec["tx"] = float(T.params[0, 2])
        rec["ty"] = float(T.params[1, 2])

        # Deterministic fit/hold-out split of the matched pairs.
        rng = np.random.default_rng(seed + i)
        order = rng.permutation(len(s))
        n_fit = max(2, int(round(len(s) * fit_frac)))
        fit_idx = order[:n_fit]
        hold_idx = order[n_fit:]
        rec["fit_n"] = int(len(fit_idx))
        rec["hold_n"] = int(len(hold_idx))

        for mname in ("euclidean", "similarity"):
            m = {}
            model = _fit(mname, s[fit_idx], t[fit_idx])
            m["fit_scale"] = float(np.hypot(model.params[0, 0], model.params[1, 0]))
            if len(hold_idx):
                rh = _resid(model, s[hold_idx], t[hold_idx])
                m["hold_rms"] = float(np.sqrt(np.mean(rh ** 2)))
                m["hold_p50"] = float(np.percentile(rh, 50))
                m["hold_p95"] = float(np.percentile(rh, 95))
                cm, em, km = _regions(t[hold_idx], centre, rmax)
                m["centre"] = float(np.mean(rh[cm])) if cm.any() else None
                m["edge"] = float(np.mean(rh[em])) if em.any() else None
                m["corner"] = float(np.mean(rh[km])) if km.any() else None
            rec[mname] = m
        rows.append(rec)

    return {
        "folder": folder,
        "n_frames": len(files),
        "reference": os.path.basename(ref_path),
        "reference_metric": float(ref_metric),
        "centre": list(centre),
        "rmax": rmax,
        "fit_frac": fit_frac,
        "seed": seed,
        "hot_pixels": hot_pixels,
        "header_bayer": header_bayer,
        "rows": rows,
    }


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "  -  "
    return f"{v:.{nd}f}"


def report(data):
    rows = data["rows"]
    ok = [r for r in rows if r["status"] == "ok"]
    fail = [r for r in rows if r["status"] != "ok"]

    lines = []
    lines.append("# M16 real-data witness — astroalign similarity scale (complementary, corrective C)\n")
    lines.append(f"folder={data['folder']}  frames={data['n_frames']}")
    lines.append(
        f"preprocessing: header BAYERPAT={'yes' if data.get('header_bayer', True) else 'GRBG hardcoded'} "
        f"+ hot-pixel correction={'yes' if data.get('hot_pixels', True) else 'no'} (threshold 3.0, neighborhood 5)"
    )
    lines.append(
        f"reference (production quality metric median/(1.4826*MAD)) = {data['reference']} "
        f"(metric {data['reference_metric']:.3f})"
    )
    lines.append(
        f"centre={tuple(round(c,1) for c in data['centre'])} px, corner radius={data['rmax']:.1f} px, "
        f"fit/hold-out={int(data['fit_frac']*100)}/{int((1-data['fit_frac'])*100)} split, master seed={data['seed']}"
    )
    lines.append(f"frames aligned OK={len(ok)} / failed={len(fail)} (out of {len(rows)} non-reference frames)\n")

    lines.append("## Per-frame table\n")
    lines.append(
        "| frame | matches | a.a. scale | rot(deg) | tx | ty | "
        "eucl hold RMS | sim hold RMS | eucl hold P95 | sim hold P95 | "
        "eucl corner | sim corner | runtime_s |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['file']} | - | - | - | - | - | - | - | - | - | - | - | - | {r['status']} |")
            continue
        e, s = r["euclidean"], r["similarity"]
        lines.append(
            f"| {r['file']} | {r['match_count']} | {_fmt(r['astroalign_scale'],6)} | "
            f"{_fmt(r['rotation_deg'],3)} | {_fmt(r['tx'],1)} | {_fmt(r['ty'],1)} | "
            f"{_fmt(e.get('hold_rms'))} | {_fmt(s.get('hold_rms'))} | "
            f"{_fmt(e.get('hold_p95'))} | {_fmt(s.get('hold_p95'))} | "
            f"{_fmt(e.get('corner'))} | {_fmt(s.get('corner'))} | {_fmt(r['runtime_s'],2)} |"
        )

    lines.append("\n## Aggregate scale statistics (astroalign returned scale) — corrected\n")
    scales = np.array([r["astroalign_scale"] for r in ok], dtype=np.float64)
    med = float(np.median(scales))
    mad = float(np.median(np.abs(scales - med)))
    rmax = float(data["rmax"])
    lines.append(f"| n | min | median | max | MAD | range |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| {len(scales)} | {scales.min():.6f} | {med:.6f} | {scales.max():.6f} | "
        f"{mad:.2e} | {scales.max()-scales.min():.2e} |"
    )
    lines.append("")
    lines.append("| statistic | value (ppm) | corner error @1101px (px) |")
    lines.append("|---|---|---|")
    stats_ppm, corner_px = scale_statistics(scales, rmax)
    label_rows = [
        ("|median(scale) - 1|", "abs_median_minus_1"),
        ("median(|scale - 1|)", "median_abs"),
        ("mean(|scale - 1|)", "mean_abs"),
    ]
    for label, key in label_rows:
        lines.append(f"| {label} | {stats_ppm[key]:.1f} | {corner_px[key]:.4f} |")

    lines.append("\n## Held-out improvement attributable to scale (euclidean − similarity, px)\n")
    imp_rms = np.array([r["euclidean"]["hold_rms"] - r["similarity"]["hold_rms"] for r in ok])
    imp_corner = np.array(
        [
            (r["euclidean"].get("corner") or 0.0) - (r["similarity"].get("corner") or 0.0)
            for r in ok
        ]
    )
    lines.append(f"| metric | min | median | max | mean |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| hold RMS improvement | {imp_rms.min():.4f} | {np.median(imp_rms):.4f} | "
        f"{imp_rms.max():.4f} | {imp_rms.mean():.4f} |"
    )
    lines.append(
        f"| corner improvement | {imp_corner.min():.4f} | {np.median(imp_corner):.4f} | "
        f"{imp_corner.max():.4f} | {imp_corner.mean():.4f} |"
    )

    lines.append("\n## Held-out residual levels (px, both models across all frames)\n")
    e_rms = np.array([r["euclidean"]["hold_rms"] for r in ok])
    s_rms = np.array([r["similarity"]["hold_rms"] for r in ok])
    lines.append("| model | median hold RMS | max hold RMS | median hold P95 | max hold P95 |")
    lines.append("|---|---|---|---|---|")
    for name, rr in (("euclidean (current)", e_rms), ("similarity", s_rms)):
        p95 = np.array([ok[i][name.split()[0]]["hold_p95"] for i in range(len(ok))])
        lines.append(f"| {name} | {np.median(rr):.4f} | {rr.max():.4f} | {np.median(p95):.4f} | {p95.max():.4f} |")

    lines.append("\n## Hold-out limitation (stated explicitly)\n")
    lines.append(
        "astroalign matched-pair selection / RANSAC ran on **all** detected stars "
        "**before** the 70/30 fit/hold-out split.  Held-out residuals are a "
        "**model-fit hold-out**, not a fully independent correspondence-selection "
        "validation: the correspondences were chosen by astroalign using every star."
    )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="/home/tristan/M16/quick")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="optional JSON output path")
    args = ap.parse_args()

    data = run(args.folder, seed=args.seed)
    print(report(data))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
