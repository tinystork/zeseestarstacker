"""REWORK-1 corrected real-engine witness run (research only).

Sections:
1. formula proof: real ZSSS pixmap == s*x + (s-1)? (empirical fit)
2. kernel x scale matrix (constant + star) with REAL pixmaps, psr=current(1)
3. psr candidates at scale 3: 1 (current), 1/s, None-estimate, s (diag)
4. true translated/rotated partial footprints (square/lanczos2/lanczos3 x3)
5. honest excursion (overshoot beyond input domain on WHT>eps)
6. transform-dither accumulation (N frames, identical scene, delta transforms)
7. iscale brightness/point-flux with area normalisation
8. mixed-exposure equation witness + nonuniform quality weighting (2 frames)
9. independent support pair + mask comparison
10. resume x3 continuous vs from_native_state (production DrizzleAccumulator)

Artifacts: rework1/artifacts/*.json/csv/npz + manifest.json (sha256).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geo import (  # noqa: E402
    ARTIFACTS,
    deposit,
    prove_formula,
    psr_none_est,
    real_pixmap,
    tf_identity,
    tf_rotate,
    tf_shift,
    zsss_output,
)
from measure import (  # noqa: E402
    aperture_photometry,
    excursion,
    scene_constant,
    scene_edge,
    scene_star,
    stats_2d,
    support_pair,
)
from seestar.core.drizzle_core import DrizzleAccumulator  # noqa: E402

ROWS = []
MANIFEST = {}


def rec(**kw):
    ROWS.append(kw)
    return kw


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_npz(key, **arrs):
    p = ARTIFACTS / f"{key}.npz"
    np.savez_compressed(p, **arrs)
    MANIFEST[str(p)] = sha(p)
    return p


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cmd = " ".join(sys.argv)
    MANIFEST["command"] = cmd

    # ---------------- 1. formula proof ----------------
    for s in (1, 2, 3, 4):
        f = prove_formula(s)
        rec(**{"row": "formula", "scale": s, **f})
        print("formula", s, f)

    # ---------------- 2. kernel x scale (constant A, star B) ----------------
    const = scene_constant((48, 48), value=100.0, noise=0.01, seed=0)
    star = scene_star((48, 48), peak=1000.0, cx=24.0, cy=24.0,
                      sigma=1.5, bg=100.0, noise=0.01, seed=1)
    for kernel in ("square", "lanczos2", "lanczos3", "point", "turbo",
                   "gaussian"):
        for s in (1, 2, 3, 4):
            for name, data in (("A", const), ("B", star)):
                sci, wht, _, _ = deposit(data, tf_identity(), s,
                                         kernel=kernel, exptime=1.0)
                st = stats_2d(sci)
                ex = excursion(sci, float(data.min()), float(data.max()), wht)
                ap = None
                if name == "B":
                    # find peak on support
                    pos = wht > 1e-9
                    idx = np.unravel_index(int(np.argmax(np.where(
                        pos, sci, -np.inf))), sci.shape)
                    ap = aperture_photometry(sci, idx[1], idx[0])
                rec(row="kxscale", kernel=kernel, scale=s, input=name,
                    sci=st, excursion=ex, star=ap)
                save_npz(f"kx_{kernel}_s{s}_{name}", sci=sci, wht=wht)

    # ---------------- 3. psr candidates scale 3 (star B) ----------------
    for kernel in ("square", "lanczos2", "lanczos3", "gaussian", "turbo"):
        for psr_tag, psr in (("current1", 1.0), ("inv_1_over_s", 1.0 / 3.0),
                             ("none", None), ("s_diag", 3.0)):
            kw = {}
            if psr is not None:
                kw["pixel_scale_ratio"] = psr
            sci, wht, _, _ = deposit(star, tf_identity(), 3, kernel=kernel,
                                     exptime=1.0, **kw)
            pos = wht > 1e-9
            idx = np.unravel_index(int(np.argmax(np.where(
                pos, sci, -np.inf))), sci.shape)
            ap = aperture_photometry(sci, idx[1], idx[0])
            ex = excursion(sci, float(star.min()), float(star.max()), wht)
            rec(row="psr", kernel=kernel, psr_tag=psr_tag, psr=psr,
                wht_min=float(wht.min()), wht_max=float(wht.max()),
                wht_neg_frac=float(np.mean(wht < 0)),
                star=ap, excursion=ex)
            save_npz(f"psr_{kernel}_{psr_tag}", sci=sci, wht=wht)
    # upstream None-estimate value for the record
    est = psr_none_est(3.0, (48, 48))
    rec(row="psr_estimate", estimate=est)

    # ---------------- 4. true translated/rotated partial footprints x3 ----
    edge = scene_edge((48, 48), level=200.0)
    for kernel in ("square", "lanczos2", "lanczos3"):
        # compose 2x3 affines via homogeneous 3x3
        def _compose(a, b):
            A = np.vstack([a, [0.0, 0.0, 1.0]])
            B = np.vstack([b, [0.0, 0.0, 1.0]])
            return (A @ B)[:2, :]

        rt = _compose(tf_rotate(25.0, 24.0, 24.0), tf_shift(4.0, 4.0))
        for tag, tf in (("translated", tf_shift(9.0, 5.0)),
                        ("rotated", tf_rotate(25.0, 24.0, 24.0)),
                        ("rot_trans", rt)):
            sci, wht, pix, mask = deposit(edge, tf, 3, kernel=kernel,
                                          exptime=1.0)
            cov = float(np.mean(wht > 1e-9))
            ex = excursion(sci, float(edge.min()), float(edge.max()), wht)
            save_npz(f"fp_{kernel}_{tag}", sci=sci, wht=wht,
                     pix_used=mask.astype(np.float32))
            rec(row="footprint", kernel=kernel, tag=tag, coverage=cov,
                wht_neg_frac=float(np.mean(wht < 0)), excursion=ex,
                sci_stats=stats_2d(sci))

    # ---------------- 5/6. transform dither accumulation ----------------
    dither_star = scene_star((48, 48), peak=1000.0, cx=24.0, cy=24.0,
                             sigma=1.2, bg=10.0, noise=0.0, seed=2)
    for kernel in ("square", "lanczos2", "lanczos3"):
        for n in (1, 2, 4, 16, 64):
            sci = wht = None
            total = 0.0
            for i in range(n):
                # identical physical scene, transform-dithered by i*0.25 px
                t = tf_shift(0.25 * i, 0.0)
                r = deposit(dither_star, t, 3, kernel=kernel, exptime=1.0)
                if sci is None:
                    sci, wht = r[0], r[1]
                else:
                    sci, wht = deposit(
                        dither_star, t, 3, kernel=kernel, exptime=1.0,
                        out_img=sci, out_wht=wht, total_exptime=float(i))[0:2]
                total += 1.0
            ex = excursion(sci, float(dither_star.min()),
                           float(dither_star.max()), wht)
            rec(row="dither", kernel=kernel, n=n, excursion=ex,
                wht_pos_min=float(np.min(wht[wht > 0])) if np.any(wht > 0)
                else None)

    # ---------------- 7. iscale: brightness vs point flux ----------------
    for s in (1, 3):
        c = scene_constant((32, 32), value=100.0, noise=0.0)
        area_ratio = s * s  # output pixels per input pixel
        for tag, isc in (("iscale1", 1.0), ("iscale_s2", float(s * s)),
                         ("iscale_1_over_s2", 1.0 / (s * s))):
            sci, wht, _, _ = deposit(c, tf_identity(), s, kernel="square",
                                     exptime=1.0, iscale=isc)
            pos = wht > 1e-9
            level = float(np.mean(sci[pos])) if np.any(pos) else None
            total_out = float(np.sum(sci[pos] * (1.0 / area_ratio)))
            rec(row="iscale", scale=s, tag=tag, iscale=isc,
                output_level=level,
                area_normalised_total=total_out,
                input_total=float(np.sum(c)))
    # point flux: total over the star divided by output area per input px
    for s in (1, 2, 3, 4):
        st = scene_star((64, 64), peak=1.0, cx=32.0, cy=32.0, sigma=1.2,
                        bg=0.0, noise=0.0)
        sci, wht, _, _ = deposit(st, tf_identity(), s, kernel="square",
                                 exptime=1.0)
        pos = wht > 1e-9
        total = float(np.sum(sci[pos])) / float(s * s)
        inp_total = float(np.sum(st))
        rec(row="pointflux_square", scale=s,
            input_total=inp_total,
            output_area_normalised=total,
            ratio=total / inp_total if inp_total else None)

    # ---------------- 8. mixed exposures + quality weighting ----------------
    # Physical scene: rate 100 cps.  counts = rate*exptime.
    for exps in ((10.0, 30.0), (10.0, 20.0, 30.0)):
        sci = wht = None
        tot = 0.0
        for t in exps:
            data = scene_constant((32, 32), value=100.0 * t, noise=0.0)
            if sci is None:
                sci, wht, _, _ = deposit(data, tf_identity(), 2,
                                         kernel="square", exptime=t)
            else:
                sci, wht, _, _ = deposit(data, tf_identity(), 2,
                                         kernel="square", exptime=t,
                                         out_img=sci, out_wht=wht,
                                         total_exptime=tot)
            tot += t
        pos = wht > 1e-9
        level = float(np.mean(sci[pos])) if np.any(pos) else None
        # exposure-weighted expectation: sum(rate_i * t_i)/sum(t_i) = 100
        rec(row="exposure_mixed", exps=list(exps), measured_rate=level,
            expected_rate=100.0)
    # weighting: two frames, different science, nonuniform weights, square x3
    w = np.ones((32, 32), dtype=np.float32)
    w[0:10, :] = 0.25
    w[20:24, 10:20] = 0.75
    f1 = scene_constant((32, 32), value=50.0, noise=0.0)
    f2 = scene_constant((32, 32), value=150.0, noise=0.0)
    sci1, wht1, _, _ = deposit(f1, tf_identity(), 3, kernel="square",
                               exptime=10.0, weight_map=w)
    sci2, wht2, _, _ = deposit(f2, tf_identity(), 3, kernel="square",
                               exptime=10.0, weight_map=w,
                               out_img=sci1, out_wht=wht1, total_exptime=10.0)
    pos = wht2 > 1e-9
    # weight image per output pixel ~ per-input weight mapped; compare only
    # regions of uniform weight
    for (y0, y1), (x0, x1), wl in (((0, 10), (0, 32), 0.25),
                                   ((24, 32), (0, 32), 1.0)):
        region = (sci2[y0:y1, x0:x1], wht2[y0:y1, x0:x1])
        m = region[1] > 1e-9
        level = float(np.mean(region[0][m])) if np.any(m) else None
        expected = (5.0 * wl * 10.0 + 15.0 * wl * 10.0) / (2 * wl * 10.0)
        rec(row="weighting", kernel="square", region=wl,
            measured_rate=level, expected_rate=expected)

    # ---------------- 9. independent support pair + masks ----------------
    st = scene_star((48, 48), peak=5000.0, cx=24.0, cy=24.0, sigma=1.2,
                    bg=10.0, noise=0.0)
    for kernel in ("square", "lanczos2", "lanczos3"):
        sci, wht, _, _ = deposit(st, tf_identity(), 3, kernel=kernel,
                                 exptime=1.0)
        sup = support_pair(st, tf_identity(), 3, exptime=1.0)
        native_mask = wht > 1e-9
        sup_mask = sup > 1e-9
        both = native_mask & sup_mask
        only_native = native_mask & ~sup_mask
        only_sup = sup_mask & ~native_mask
        rec(row="support", kernel=kernel,
            native_pos_frac=float(np.mean(native_mask)),
            sup_pos_frac=float(np.mean(sup_mask)),
            both_frac=float(np.mean(both)),
            native_not_sup_frac=float(np.mean(only_native)),
            sup_not_native_frac=float(np.mean(only_sup)),
            n_eff=float(np.sum(sup_mask)),
            sci_max_both=float(np.max(np.abs(sci[both]))) if np.any(both)
            else None,
            sci_max_native_only=float(np.max(np.abs(sci[only_native])))
            if np.any(only_native) else None)
        save_npz(f"sup_{kernel}", sci=sci, wht=wht, sup=sup)

    # ---------------- 10. resume x3 via production accumulator seam --------
    shape = (32, 32)
    star_r = scene_star(shape, peak=1000.0, cx=16.0, cy=16.0, sigma=1.2,
                        bg=5.0, noise=0.0)
    out_shape_hw = (int(round(shape[0] * 3)), int(round(shape[1] * 3)))
    for kernel in ("square", "lanczos2"):
        # continuous: production accumulator, 3 adds
        acc = DrizzleAccumulator(out_shape_hw, kernel=kernel, pixfrac=1.0,
                                 fillval="0.0")
        pix, mask, _, _, _ = real_pixmap(shape, tf_identity(), 3)
        wm = np.ones(shape, dtype=np.float32) * mask.astype(np.float32)
        for i in range(3):
            acc.add(star_r, wm, pix, exptime=5.0, in_units="counts")
        sci_c = np.array(acc._out_img, copy=True)
        wht_c = np.array(acc._out_wht, copy=True)
        # resume: 1 add, snapshot, from_native_state, 2 adds
        acc1 = DrizzleAccumulator(out_shape_hw, kernel=kernel, pixfrac=1.0,
                                  fillval="0.0")
        acc1.add(star_r, wm, pix, exptime=5.0, in_units="counts")
        acc2 = DrizzleAccumulator.from_native_state(
            out_shape_hw, np.array(acc1._out_img, copy=True),
            np.array(acc1._out_wht, copy=True), kernel=kernel,
            pixfrac=1.0, fillval="0.0", total_exptime=5.0)
        for i in range(2):
            acc2.add(star_r, wm, pix, exptime=5.0, in_units="counts")
        rec(row="resume_x3", kernel=kernel,
            sci_equal=bool(np.array_equal(sci_c, acc2._out_img)),
            wht_equal=bool(np.array_equal(wht_c, acc2._out_wht)),
            max_diff=float(np.max(np.abs(sci_c - acc2._out_img))),
            total_exptime=acc2._total_exptime)
        save_npz(f"resume_{kernel}_x3", sci_c=sci_c, wht_c=wht_c,
                 sci_r=np.array(acc2._out_img, copy=True),
                 wht_r=np.array(acc2._out_wht, copy=True))

    # ---------------- write outputs + manifest ----------------
    with open(ARTIFACTS / "rows.json", "w") as fh:
        json.dump(ROWS, fh, indent=1)
    # fixed union schema CSV (all row types preserved)
    keys = []
    for r in ROWS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(ARTIFACTS / "rows.csv", "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=keys)
        wcsv.writeheader()
        for r in ROWS:
            row = {}
            for k in keys:
                v = r.get(k)
                row[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
            wcsv.writerow(row)
    manifest = {"command": cmd,
                "files": {str(p): sha(p)
                          for p in sorted(ARTIFACTS.iterdir())
                          if p.is_file()}}
    with open(ARTIFACTS / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"done: {len(ROWS)} rows -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
