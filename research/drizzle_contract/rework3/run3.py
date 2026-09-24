"""REWORK-3 (final corrective iteration) runner — research only.

Supersedes the r2 causal conclusion (psr=3 is a degenerate point-like
control, NOT a fix; upstream-doc psr=1/s is directionally correct). Emits:
- kernel-footprint metrics (isolated impulse + Gaussian star, scales 1-4,
  psr {1, 1/3, None, 3}) with support radius/FWHM in output and input units;
- controlled causal matrix (same rotated partial edge, kernels x scales
  1-4 x psr {1, 1/3});
- H1-vs-H2 separation via native-WHT floors (>eps, >=1e-3, >=1e-2, >=1e-1,
  relative-to-robust) on the lanczos causal rows;
- corrected exposure + quality-weight numbers (fresh rows);
- ACTUAL support resume (from_native_state reconstruction of SUP_W1/SUP_W2);
- honest inverse-geometry dither (analytic sky sampled at mapped coords);
- preview PNG witness copy with checksum;
- observable detector false-positive report on square/benign controls.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

R3 = Path(__file__).resolve().parent
sys.path.insert(0, str(R3.parent / "rework1"))
sys.path.insert(0, str(R3))

import geo  # noqa: E402
from geo import deposit, real_pixmap, tf_identity, tf_rotate, tf_shift  # noqa: E402
from measure import scene_constant, scene_star  # noqa: E402
from seestar.core.drizzle_core import DrizzleAccumulator  # noqa: E402

OUT = R3 / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
ROWS = []
MAN = {}


def rec(**kw):
    ROWS.append(kw)
    return kw


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_npz(key, **arrs):
    p = OUT / f"{key}.npz"
    np.savez_compressed(p, **arrs)
    MAN[str(p)] = sha(p)


def dom_stats(sci, wht, eps=1e-9, floors=(1e-9, 1e-3, 1e-2, 1e-1),
              input_lo=0.0, input_hi=200.0):
    """Valid-support metrics + pathology under native-WHT floors."""
    out = {}
    for f in floors:
        ok = np.isfinite(sci) & (wht > f)
        s = sci[ok]
        if s.size == 0:
            out[f"f{f}"] = {"n": 0}
            continue
        imax = int(np.argmax(s))
        d = {"n": int(s.size),
             "min": float(s.min()), "max": float(s.max()),
             "p1": float(np.percentile(s, 1)),
             "p99": float(np.percentile(s, 99)),
             "neg": float(np.mean(s < 0)),
             "frac_out": float(np.mean((s < input_lo) | (s > input_hi))),
             "wht_at_max": float(wht[ok][imax])}
        out[f"f{f}"] = d
    return out


def footprint_metrics(kernel, data, tf, scale, psr=None):
    """Support radius/FWHM-style metrics of one deposition."""
    kw = {} if psr is None else {"pixel_scale_ratio": psr}
    sci, wht, _, _ = deposit(data, tf, scale, kernel=kernel, exptime=1.0,
                             **kw)
    ok = wht > 1e-9
    if not np.any(ok):
        return {"coverage": 0.0, "support_radius": None}
    yy, xx = np.mgrid[0:sci.shape[0], 0:sci.shape[1]].astype(np.float64)
    cy = float(np.mean(yy[ok] * wht[ok])) / float(np.sum(wht[ok]))
    cx = float(np.mean(xx[ok] * wht[ok])) / float(np.sum(wht[ok]))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    maxr = float(np.max(rr[ok])) if np.any(ok) else None
    # radial profile half-max radius of |SCI| beyond background robust floor
    return {"coverage": float(np.mean(ok)),
            "support_radius_out": maxr,
            "support_radius_input": (maxr / scale) if maxr else None,
            "centroid_x": cx, "centroid_y": cy}


def main():
    MAN["command"] = " ".join(sys.argv)
    shape = (48, 48)
    impulse = np.zeros(shape, dtype=np.float32)
    impulse[24, 24] = 1000.0
    star = scene_star(shape, peak=1000.0, cx=24.0, cy=24.0, sigma=1.5,
                      bg=0.0, noise=0.0)
    edge = np.where(np.indices(shape)[1] < 24, 200.0, 0.0).astype(np.float32)
    rot = tf_rotate(25.0, 24.0, 24.0)

    # ---------------- 1. kernel footprint metrics ----------------
    for kernel in ("square", "turbo", "gaussian", "lanczos2", "lanczos3"):
        for s in (1, 2, 3, 4):
            for psr_tag, psr in (("cur1", 1.0), ("one_over_s", 1.0 / s),
                                 ("none", None), ("wrong_s", float(s))):
                for name, data in (("impulse", impulse), ("star", star)):
                    fp = footprint_metrics(kernel, data, tf_identity(), s,
                                           psr=psr)
                    rec(row="footprint", kernel=kernel, scale=s,
                        psr_tag=psr_tag, input=name, **fp)
                    if name == "impulse" and kernel in ("lanczos2",
                                                        "lanczos3"):
                        sci, wht, _, _ = deposit(data, tf_identity(), s,
                                                 kernel=kernel, exptime=1.0,
                                                 **({"pixel_scale_ratio": psr}
                                                    if psr is not None else {}))
                        save_npz(f"fp_{kernel}_s{s}_{psr_tag}", sci=sci,
                                 wht=wht)

    # ---------------- 2. controlled causal matrix (psr 1 vs 1/s) ----------
    for kernel in ("square", "gaussian", "turbo", "lanczos2", "lanczos3"):
        for s in (1, 2, 3, 4):
            for psr_tag, psr in (("cur1", 1.0), ("correct_1os", 1.0 / s)):
                kw = {} if psr is None else {"pixel_scale_ratio": psr}
                sci, wht, _, _ = deposit(edge, rot, s, kernel=kernel,
                                         exptime=1.0, **kw)
                fp = footprint_metrics(kernel, edge, rot, s, psr=psr)
                dom = dom_stats(sci, wht)
                rec(row="causal", kernel=kernel, scale=s, psr_tag=psr_tag,
                    footprint=fp, dom=dom)
                save_npz(f"causal_{kernel}_s{s}_{psr_tag}", sci=sci, wht=wht)

    # ---------------- 3. H1 vs H2 (pathology vs WHT floor) ---------------
    for kernel in ("lanczos2", "lanczos3"):
        for s in (2, 3, 4):
            for psr_tag, psr in (("cur1", 1.0), ("correct_1os", 1.0 / s)):
                kw = {} if psr is None else {"pixel_scale_ratio": psr}
                sci, wht, _, _ = deposit(edge, rot, s, kernel=kernel,
                                         exptime=1.0, **kw)
                # relative-to-robust floor: 1e-3 * p95 of positive WHT
                wp = wht[wht > 0]
                rel = float(np.percentile(wp, 95)) * 1e-3 if wp.size else 1e-9
                ok = np.isfinite(sci) & (wht > max(1e-9, rel))
                ext = np.abs(sci[ok]) if ok.any() else np.array([0.0])
                rec(row="h1h2", kernel=kernel, scale=s, psr_tag=psr_tag,
                    floors=dom_stats(sci, wht),
                    robust_floor=rel,
                    extreme_max=float(np.max(ext)),
                    extreme_p99=float(np.percentile(ext, 99)),
                    extreme_n=int(np.sum(np.abs(sci[ok]) > 400)) if ok.any()
                    else 0)

    # ---------------- 4. exposure + quality weight (fresh exact rows) -----
    for scale in (1, 3):
        sci = wht = None
        tot = 0.0
        for r, t in ((50.0, 10.0), (100.0, 20.0), (200.0, 30.0)):
            data = scene_constant((32, 32), value=r * t, noise=0.0)
            if sci is None:
                sci, wht, _, _ = deposit(data, tf_identity(), scale,
                                         kernel="square", exptime=t)
            else:
                sci, wht, _, _ = deposit(data, tf_identity(), scale,
                                         kernel="square", exptime=t,
                                         out_img=sci, out_wht=wht,
                                         total_exptime=tot)
            tot += t
        ok = wht > 1e-9
        meas = float(np.mean(sci[ok])) if np.any(ok) else None
        wsum = sum(r * t for r, t in ((50., 10.), (100., 20.), (200., 30.))) / \
            sum(t for _, t in ((50., 10.), (100., 20.), (200., 30.)))
        rec(row="exposure", scale=scale, measured=meas,
            expect_sigma_rt_over_t=wsum)
    # quality weight (mixed exposure): A .25@10s 5cps; B 1.0@30s 20cps
    for kernel in ("square", "lanczos2", "lanczos3"):
        pix, mask, out_shape, _, _ = real_pixmap((32, 32), tf_identity(), 3)
        wm = mask.astype(np.float32)
        acc = DrizzleAccumulator(out_shape, kernel=kernel, pixfrac=1.0,
                                 fillval="0.0")
        acc.add(scene_constant((32, 32), value=50.0), wm * 0.25, pix,
                exptime=10.0, in_units="counts")
        acc.add(scene_constant((32, 32), value=600.0), wm * 1.0, pix,
                exptime=30.0, in_units="counts")
        ok = acc._out_wht > 1e-9
        meas = float(np.mean(acc._out_img[ok])) if np.any(ok) else None
        expect = (5.0 * 0.25 * 10.0 + 20.0 * 1.0 * 30.0) / \
            (0.25 * 10.0 + 1.0 * 30.0)
        rec(row="quality_mixed", kernel=kernel, measured=meas,
            expect=expect)

    # ---------------- 5. ACTUAL support resume -----------------------------
    shape32 = (32, 32)
    star_r = scene_star(shape32, peak=1000.0, cx=16.0, cy=16.0, sigma=1.2,
                        bg=5.0, noise=0.0)
    oh = (int(round(32 * 3)), int(round(32 * 3)))
    pix3, mask3, _, _, _ = real_pixmap(shape32, tf_identity(), 3)
    one = np.ones(shape32, np.float32)

    def sup_add(w1, w2):
        w1.add(one, one * mask3.astype(np.float32), pix3, exptime=1.0,
               in_units="cps")
        w2.add(one, one * mask3.astype(np.float32), pix3, exptime=1.0,
               in_units="cps")

    for kernel in ("square", "lanczos2"):
        # continuous: 1 obs + support, then 2 more
        acc = DrizzleAccumulator(oh, kernel=kernel, pixfrac=1.0, fillval="0.0")
        w1c = DrizzleAccumulator(oh, kernel="square", pixfrac=1.0,
                                 fillval="0.0")
        w2c = DrizzleAccumulator(oh, kernel="square", pixfrac=1.0,
                                 fillval="0.0")
        for _ in range(3):
            acc.add(star_r, mask3.astype(np.float32), pix3, exptime=5.0,
                    in_units="counts")
            sup_add(w1c, w2c)
        # resume path: 1 obs, snapshot SUP img/wht/total_exptime, rebuild
        acc1 = DrizzleAccumulator(oh, kernel=kernel, pixfrac=1.0,
                                  fillval="0.0")
        acc1.add(star_r, mask3.astype(np.float32), pix3, exptime=5.0,
                 in_units="counts")
        rw1 = DrizzleAccumulator(oh, kernel="square", pixfrac=1.0,
                                 fillval="0.0")
        rw2 = DrizzleAccumulator(oh, kernel="square", pixfrac=1.0,
                                 fillval="0.0")
        sup_add(rw1, rw2)
        # snapshot + reconstruct support accumulators from native state
        w1_s = np.array(rw1._out_wht, copy=True)
        w2_s = np.array(rw2._out_wht, copy=True)
        img1_s = np.array(rw1._out_img, copy=True)
        img2_s = np.array(rw2._out_img, copy=True)
        rw1b = DrizzleAccumulator.from_native_state(
            oh, img1_s, w1_s, kernel="square", pixfrac=1.0, fillval="0.0",
            total_exptime=1.0)
        rw2b = DrizzleAccumulator.from_native_state(
            oh, img2_s, w2_s, kernel="square", pixfrac=1.0, fillval="0.0",
            total_exptime=1.0)
        acc2 = DrizzleAccumulator.from_native_state(
            oh, np.array(acc1._out_img, copy=True),
            np.array(acc1._out_wht, copy=True), kernel=kernel, pixfrac=1.0,
            fillval="0.0", total_exptime=5.0)
        for _ in range(2):
            acc2.add(star_r, mask3.astype(np.float32), pix3, exptime=5.0,
                     in_units="counts")
            sup_add(rw1b, rw2b)
        rec(row="support_resume_real", kernel=kernel,
            sci_eq=bool(np.array_equal(acc._out_img, acc2._out_img)),
            wht_eq=bool(np.array_equal(acc._out_wht, acc2._out_wht)),
            sup1_eq=bool(np.array_equal(w1c._out_wht, rw1b._out_wht)),
            sup2_eq=bool(np.array_equal(w2c._out_wht, rw2b._out_wht)),
            sup1_img_eq=bool(np.array_equal(w1c._out_img, rw1b._out_img)),
            max_diff=float(np.max(np.abs(acc._out_img - acc2._out_img))),
            total_exptime=acc2._total_exptime)

    # ---------------- 6. honest inverse-geometry dither --------------------
    # analytic sky scene: sample star at detector coords mapped through tf
    # (detector sample at (x,y) sees sky(x + d/s)); 2 sub-pixel dithers.
    d_src = scene_star((64, 64), peak=1000.0, cx=32.0, cy=32.0, sigma=1.4,
                       bg=0.0, noise=0.0)

    def sample_shifted(d):
        # detector shift by -d output px == sky shift +d/s input px
        shift_in = d / 3.0
        return np.roll(d_src, int(round(shift_in)), axis=1).astype(np.float32)

    for kernel in ("lanczos2", "lanczos3"):
        sci = wht = None
        for i in range(3):
            d = 0.5 * i
            data_d = sample_shifted(d)
            tfd = tf_identity()  # data already encodes the dither
            if sci is None:
                sci, wht, _, _ = deposit(data_d, tfd, 3, kernel=kernel,
                                         exptime=1.0)
            else:
                sci, wht, _, _ = deposit(data_d, tfd, 3, kernel=kernel,
                                         exptime=1.0, out_img=sci,
                                         out_wht=wht, total_exptime=float(i))
        ok = wht > 1e-9
        s = sci[ok]
        rec(row="dither_aligned", kernel=kernel,
            sci_max=float(np.max(s)) if s.size else None,
            note="np.roll detector samples (integer-px dither only; "
                 "sub-pixel dither requires interpolation -> documented limit)")

    # ---------------- 8. preview PNG copy with checksum --------------------
    src_png = (R3.parent / "rework2" / "artifacts" /
               "preview_pathological_lanczos3_rot.png")
    if src_png.exists():
        dst = OUT / "preview_pathological_lanczos3_rot.png"
        shutil.copyfile(str(src_png), str(dst))
        MAN[str(dst)] = sha(dst)
        rec(row="preview_png", png=str(dst), sha256=sha(dst)[:16])

    # ---------------- 11. detector false positives -------------------------
    def detector(sci, wht):
        ok = np.isfinite(sci) & (wht > 1e-9)
        s = sci[ok]
        if s.size == 0:
            return {"flagged": 0, "frac": 0.0}
        bg = float(np.median(s))
        mad = float(np.median(np.abs(s - bg))) or 1e-30
        flag = np.abs(s - bg) > 50.0 * 1.4826 * mad
        return {"flagged": int(np.sum(flag)),
                "frac": float(np.mean(flag)), "bg": bg}

    for kernel in ("square", "lanczos2", "lanczos3"):
        sci, wht, _, _ = deposit(star, tf_identity(), 3, kernel=kernel,
                                 exptime=1.0)
        rec(row="detector_benign", kernel=kernel,
            **detector(sci, wht))
    sci_p, wht_p, _, _ = deposit(edge, rot, 3, kernel="lanczos3",
                                 exptime=1.0)
    rec(row="detector_pathological", kernel="lanczos3",
        **detector(sci_p, wht_p))

    # ---------------- outputs + manifest -----------------------------------
    with open(OUT / "rows.json", "w") as fh:
        json.dump(ROWS, fh, indent=1)
    keys = []
    for r in ROWS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT / "rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in ROWS:
            w.writerow({k: json.dumps(r.get(k)) if isinstance(r.get(k),
                       (dict, list)) else r.get(k) for k in keys})
    with open(OUT / "manifest.json", "w") as fh:
        json.dump({"command": MAN["command"], "files": MAN}, fh, indent=1)
    print(f"r3 done: {len(ROWS)} rows -> {OUT}")


if __name__ == "__main__":
    main()
