"""REWORK-2 comprehensive runner (research only).

Executes the bounded required fixes against the REAL drizzle 2.2.0 engine
with real ZSSS pixmap geometry (rework1 helpers, imported immutably):

1. causal psr A/B: same scenes x kernels x {psr 1.0, 1/3, None, 3} scale 3;
   lanczos2/3 rotated partial x scales 1-4 (metrics on WHT>eps only).
2. iscale {1, 1/s^2, s^2} constant + isolated source x scales 1-4
   (raw sum + area-normalised integral).
3. exposure estimator: rates 50@10,100@20,200@30 cps (square x1/x3) vs the
   two candidate expectations.
4. nontrivial quality weight: frame A weight .25, frame B 1.0 (same t, then
   mixed t) square/lanczos2/lanczos3 x3.
5. real support pair SUP_W1/SUP_W2/N_eff (M3 semantics) incl. pathological
   rotated cells; candidate masks (robust WHT stability + positive support).
6. support resume x3 continuous vs from_native_state (SCI/WHT + support).
7. repeat N=1/2/4/16/64 identical data+pixmap (authoritative) and an
   inverse-geometry dither witness.
8. preview PNG witness of the pathological rotated lanczos3 output through
   the production display stretch.
12. boundary detector on observables (no input-domain clipping); false
    positives tested on square controls + benign lanczos full coverage.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

R2 = Path(__file__).resolve().parent
sys.path.insert(0, str(R2.parent / "rework1"))
sys.path.insert(0, str(R2))

import geo  # noqa: E402
import measure as M  # noqa: E402

from geo import deposit, real_pixmap, tf_identity, tf_rotate, tf_shift  # noqa: E402
from measure import scene_constant, scene_star  # noqa: E402
from seestar.core.drizzle_core import DrizzleAccumulator  # noqa: E402

OUT = R2 / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
ROWS = []
MAN = {}


def rec(**kw):
    ROWS.append(kw)
    return kw


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npz(key, **arrs):
    p = OUT / f"{key}.npz"
    np.savez_compressed(p, **arrs)
    MAN[str(p)] = sha(p)


def dom_stats(sci, wht, eps=1e-9):
    """Metrics on finite WHT>eps support only."""
    ok = np.isfinite(sci) & (wht > eps) & np.isfinite(wht)
    s = sci[ok]
    w = wht[ok]
    out = {"n": int(s.size)}
    if s.size == 0:
        return out
    out.update({
        "min": float(s.min()), "max": float(s.max()),
        "p1": float(np.percentile(s, 1)), "p5": float(np.percentile(s, 5)),
        "p95": float(np.percentile(s, 95)), "p99": float(np.percentile(s, 99)),
        "neg_frac": float(np.mean(s < 0)),
    })
    imin, imax = int(np.argmin(s)), int(np.argmax(s))
    out["wht_at_min"] = float(w[imin])
    out["wht_at_max"] = float(w[imax])
    return out


def edge_scene():
    return M.scene_edge((48, 48), level=200.0)


def rotated_tf():
    return tf_rotate(25.0, 24.0, 24.0)


def _add_support_observation(acc_w1, acc_w2, s_i, w_i, pix, mask):
    """Production M3 support semantics per observation s_i (validity mask)."""
    acc_w1.add(np.asarray(s_i, dtype=np.float32),
               np.asarray(s_i, dtype=np.float32) * mask.astype(np.float32),
               pix, exptime=1.0, in_units="cps")
    acc_w2.add(np.asarray(s_i, dtype=np.float32),
               np.asarray(s_i, dtype=np.float32) ** 2 * mask.astype(np.float32),
               pix, exptime=1.0, in_units="cps")


def main():
    cmd = " ".join(sys.argv)
    MAN["command"] = cmd
    shape = (48, 48)
    star = scene_star(shape, peak=1000.0, cx=24.0, cy=24.0, sigma=1.5,
                      bg=100.0, noise=0.01, seed=1)
    edge = edge_scene()

    # ---------------- 1. causal psr A/B ----------------
    scenes = {"translated": tf_shift(9.0, 5.0),
              "rotated": rotated_tf(),
              "rot_trans": None}
    # compose rot+trans properly
    A = np.vstack([rotated_tf(), [0, 0, 1.0]])
    B = np.vstack([tf_shift(4.0, 4.0), [0, 0, 1.0]])
    scenes["rot_trans"] = (A @ B)[:2, :]
    kernels = ("square", "gaussian", "turbo", "lanczos2", "lanczos3")
    for k in kernels:
        for tag, tf in scenes.items():
            for psr_tag, psr in (("cur1", 1.0), ("inv13", 1.0 / 3.0),
                                 ("none", None), ("diag3", 3.0)):
                kw = {} if psr is None else {"pixel_scale_ratio": psr}
                sci, wht, _, _ = deposit(edge, tf, 3, kernel=k, exptime=1.0,
                                         **kw)
                st = dom_stats(sci, wht)
                st["input_min"] = 0.0
                st["input_max"] = 200.0
                st["frac_outside"] = float(np.mean(
                    (sci < 0) | (sci > 200))) if st["n"] else None
                rec(row="causal", kernel=k, tag=tag, psr_tag=psr_tag,
                    psr=psr, stats=st)
                save_npz(f"causal_{k}_{tag}_{psr_tag}", sci=sci, wht=wht)
    # lanczos2/3 rotated partial x scales 1..4
    for k in ("lanczos2", "lanczos3"):
        for s in (1, 2, 3, 4):
            sci, wht, _, _ = deposit(edge, rotated_tf(), s, kernel=k,
                                     exptime=1.0)
            st = dom_stats(sci, wht)
            st["input_min"] = 0.0
            st["input_max"] = 200.0
            rec(row="causal_scale", kernel=k, scale=s, stats=st)
            save_npz(f"causal_{k}_rot_s{s}", sci=sci, wht=wht)

    # ---------------- 2. iscale ----------------
    const = scene_constant((32, 32), value=100.0)
    iso = scene_star((64, 64), peak=1.0, cx=32.0, cy=32.0, sigma=1.2, bg=0.0,
                     noise=0.0)
    for s in (1, 2, 3, 4):
        for tag, isc in (("i1", 1.0), ("i_1os2", 1.0 / (s * s)),
                         ("i_s2", float(s * s))):
            for name, data in (("const", const), ("iso", iso)):
                sci, wht, _, _ = deposit(data, tf_identity(), s,
                                         kernel="square", exptime=1.0,
                                         iscale=isc)
                ok = wht > 1e-9
                raw_sum = float(np.sum(sci[ok])) if np.any(ok) else 0.0
                area_norm = raw_sum / float(s * s)
                level = float(np.mean(sci[ok])) if np.any(ok) else None
                rec(row="iscale", scale=s, tag=tag, input=name,
                    raw_sum=raw_sum, area_norm=area_norm, level=level,
                    input_sum=float(np.sum(data)))

    # ---------------- 3. exposure estimator ----------------
    rates = [(50.0, 10.0), (100.0, 20.0), (200.0, 30.0)]
    for scale in (1, 3):
        sci = wht = None
        tot = 0.0
        for r, t in rates:
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
        wsum_t = sum(r * t for r, t in rates) / sum(t for _, t in rates)
        wsum_t2 = sum(r * t * t for r, t in rates) / sum(
            t * t for _, t in rates)
        rec(row="exposure_estimator", scale=scale, measured=meas,
            expect_wt=wsum_t, expect_wt2=wsum_t2)

    # ---------------- 4. nontrivial quality weight ----------------
    pix, mask, out_shape, _, _ = real_pixmap(shape, tf_identity(), 3)
    wm_mask = mask.astype(np.float32)
    fA = scene_constant(shape, value=50.0)      # 5 cps @10s
    fB = scene_constant(shape, value=200.0)     # 20 cps @10s
    for kernel in ("square", "lanczos2", "lanczos3"):
        # same exposure 10s, per-frame weights .25 / 1.0
        acc = DrizzleAccumulator(out_shape, kernel=kernel, pixfrac=1.0,
                                 fillval="0.0")
        acc.add(fA, wm_mask * 0.25, pix, exptime=10.0, in_units="counts")
        acc.add(fB, wm_mask * 1.0, pix, exptime=10.0, in_units="counts")
        ok = acc._out_wht > 1e-9
        meas = float(np.mean(acc._out_img[ok])) if np.any(ok) else None
        # engine estimator hypothesis: rate-weighted-by (w*t)?  w's differ
        # -> expectation only valid if weights enter once (see report eq.)
        rec(row="quality_weight", kernel=kernel, exposure="same",
            measured=meas)
        # mixed exposure: A .25@10s(5cps), B 1.0@30s(20cps)
        acc2 = DrizzleAccumulator(out_shape, kernel=kernel, pixfrac=1.0,
                                  fillval="0.0")
        fA2 = scene_constant(shape, value=50.0)
        fB2 = scene_constant(shape, value=600.0)  # 20cps*30s
        acc2.add(fA2, wm_mask * 0.25, pix, exptime=10.0, in_units="counts")
        acc2.add(fB2, wm_mask * 1.0, pix, exptime=30.0, in_units="counts")
        ok2 = acc2._out_wht > 1e-9
        meas2 = float(np.mean(acc2._out_img[ok2])) if np.any(ok2) else None
        rec(row="quality_weight", kernel=kernel, exposure="mixed",
            measured=meas2)

    # ---------------- 5. real support pair + masks ----------------
    # pathological rotated lanczos3 edge cell + benign full-coverage star
    cells = []
    for kernel in ("lanczos2", "lanczos3"):
        sci, wht, _, _ = deposit(edge, rotated_tf(), 3, kernel=kernel,
                                 exptime=1.0)
        cells.append((f"path_{kernel}", sci, wht, kernel, "rot_edge"))
    for kernel in ("square", "lanczos2", "lanczos3"):
        sci, wht, _, _ = deposit(star, tf_identity(), 3, kernel=kernel,
                                 exptime=1.0)
        cells.append((f"full_{kernel}", sci, wht, kernel, "full_star"))
    for key, sci, wht, kernel, tag in cells:
        acc_w1 = DrizzleAccumulator((int(round(48 * 3)), int(round(48 * 3))),
                                    kernel="square", pixfrac=1.0,
                                    fillval="0.0")
        acc_w2 = DrizzleAccumulator((int(round(48 * 3)), int(round(48 * 3))),
                                    kernel="square", pixfrac=1.0,
                                    fillval="0.0")
        # deposit a unit-validity support observation on this geometry
        s_i = np.ones((48, 48), dtype=np.float32)
        pix2, mask2, _, _, _ = real_pixmap((48, 48), tf_identity(), 3)
        if tag == "rot_edge":
            pix2, mask2, _, _, _ = real_pixmap((48, 48), rotated_tf(), 3)
        _add_support_observation(acc_w1, acc_w2, s_i, s_i, pix2, mask2)
        sup1 = np.array(acc_w1._out_wht, copy=True)
        sup2 = np.array(acc_w2._out_wht, copy=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_eff = np.where(sup2 > 0, sup1 * sup1 / sup2, 0.0)
        save_npz(f"sup_{key}", sci=sci, wht=wht, sup1=sup1, sup2=sup2,
                 n_eff=n_eff)
        # candidate masks (observables only)
        ok_native = wht > 1e-9
        ok_sup = sup1 > 1e-9
        wht_pos = np.where(wht > 0, wht, 0.0)
        robust = np.percentile(wht_pos[wht_pos > 0], 90) if np.any(
            wht_pos > 0) else 0.0
        ok_stable = ok_native & (wht > max(1e-9, robust * 1e-3))
        dom = dom_stats(sci, wht)
        rec(row="support_masks", key=key, tag=tag,
            n_native=int(np.sum(ok_native)),
            n_sup=int(np.sum(ok_sup)),
            n_stable=int(np.sum(ok_stable)),
            robust_wht=float(robust),
            dom=dom)
    # n_eff for the pathological cell with nonuniform multi-frame support
    # (3 frames, weights .25/.5/1.0)
    acc_w1 = DrizzleAccumulator((144, 144), kernel="square", pixfrac=1.0,
                                fillval="0.0")
    acc_w2 = DrizzleAccumulator((144, 144), kernel="square", pixfrac=1.0,
                                fillval="0.0")
    pix2, mask2, _, _, _ = real_pixmap((48, 48), rotated_tf(), 3)
    for wgt in (0.25, 0.5, 1.0):
        s_i = np.ones((48, 48), dtype=np.float32) * wgt
        _add_support_observation(acc_w1, acc_w2, s_i, s_i, pix2, mask2)
    sup1 = np.array(acc_w1._out_wht, copy=True)
    sup2 = np.array(acc_w2._out_wht, copy=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(sup2 > 0, sup1 * sup1 / sup2, 0.0)
    save_npz("sup_multiframe_rot", sup1=sup1, sup2=sup2, n_eff=n_eff)
    rec(row="support_multiframe", key="rot3frames",
        n_eff_max=float(np.nanmax(n_eff)) if n_eff.size else None)

    # ---------------- 6. support resume x3 ----------------
    shape32 = (32, 32)
    star_r = scene_star(shape32, peak=1000.0, cx=16.0, cy=16.0, sigma=1.2,
                        bg=5.0, noise=0.0)
    oh = (int(round(shape32[0] * 3)), int(round(shape32[1] * 3)))
    pix3, mask3, _, _, _ = real_pixmap(shape32, tf_identity(), 3)
    for kernel in ("square", "lanczos2"):
        # continuous
        acc = DrizzleAccumulator(oh, kernel=kernel, pixfrac=1.0, fillval="0.0")
        w1 = DrizzleAccumulator(oh, kernel="square", pixfrac=1.0, fillval="0.0")
        w2 = DrizzleAccumulator(oh, kernel="square", pixfrac=1.0, fillval="0.0")
        for _ in range(3):
            acc.add(star_r, mask3.astype(np.float32), pix3, exptime=5.0,
                    in_units="counts")
            _add_support_observation(w1, w2, np.ones(shape32, np.float32),
                                     np.ones(shape32, np.float32), pix3, mask3)
        sc, wc = np.array(acc._out_img, copy=True), np.array(
            acc._out_wht, copy=True)
        s1c, s2c = np.array(w1._out_wht, copy=True), np.array(
            w2._out_wht, copy=True)
        # resume
        acc1 = DrizzleAccumulator(oh, kernel=kernel, pixfrac=1.0,
                                  fillval="0.0")
        acc1.add(star_r, mask3.astype(np.float32), pix3, exptime=5.0,
                 in_units="counts")
        acc2 = DrizzleAccumulator.from_native_state(
            oh, np.array(acc1._out_img, copy=True),
            np.array(acc1._out_wht, copy=True), kernel=kernel,
            pixfrac=1.0, fillval="0.0", total_exptime=5.0)
        rw1 = DrizzleAccumulator(oh, kernel="square", pixfrac=1.0,
                                 fillval="0.0")
        rw2 = DrizzleAccumulator(oh, kernel="square", pixfrac=1.0,
                                 fillval="0.0")
        _add_support_observation(rw1, rw2, np.ones(shape32, np.float32),
                                 np.ones(shape32, np.float32), pix3, mask3)
        # continue support accumulators after 1 observation
        for _ in range(2):
            acc2.add(star_r, mask3.astype(np.float32), pix3, exptime=5.0,
                     in_units="counts")
            _add_support_observation(rw1, rw2, np.ones(shape32, np.float32),
                                     np.ones(shape32, np.float32), pix3, mask3)
        rec(row="support_resume", kernel=kernel,
            sci_eq=bool(np.array_equal(sc, acc2._out_img)),
            wht_eq=bool(np.array_equal(wc, acc2._out_wht)),
            sup1_eq=bool(np.array_equal(s1c, rw1._out_wht)),
            sup2_eq=bool(np.array_equal(s2c, rw2._out_wht)),
            total_exptime=acc2._total_exptime,
            max_diff=float(np.max(np.abs(sc - acc2._out_img))))

    # ---------------- 7. repeat identical + inverse-geometry dither --------
    rep_frame = scene_star((32, 32), peak=500.0, cx=16.0, cy=16.0, sigma=1.2,
                           bg=5.0, noise=0.0)
    for kernel in ("square", "lanczos2", "lanczos3"):
        for n in (1, 2, 4, 16, 64):
            sci = wht = None
            for _ in range(n):
                if sci is None:
                    sci, wht, _, _ = deposit(rep_frame, tf_identity(), 3,
                                             kernel=kernel, exptime=1.0)
                else:
                    sci, wht, _, _ = deposit(
                        rep_frame, tf_identity(), 3, kernel=kernel,
                        exptime=1.0, out_img=sci, out_wht=wht,
                        total_exptime=float(_))
            ok = wht > 1e-9
            rec(row="repeat", kernel=kernel, n=n,
                sci_max=float(np.max(sci[ok])) if np.any(ok) else None,
                wht_max=float(np.max(wht)))
            save_npz(f"repeat_{kernel}_n{n}", sci=sci, wht=wht)
    # inverse-geometry dither: shift detector samples by -d so the same sky
    # source lands on the SAME output location (scene resampled consistently)
    dither_src = scene_star((64, 64), peak=500.0, cx=24.0, cy=24.0,
                            sigma=1.4, bg=5.0, noise=0.0)
    for kernel in ("lanczos2", "lanczos3"):
        sci = wht = None
        for i in range(4):
            d = 0.5 * i  # output-pixel dither; inverse: shift detector by -d/s
            tf_inv = tf_shift(-d / 3.0, 0.0)
            if sci is None:
                sci, wht, _, _ = deposit(dither_src, tf_inv, 3,
                                         kernel=kernel, exptime=1.0)
            else:
                sci, wht, _, _ = deposit(dither_src, tf_inv, 3, kernel=kernel,
                                         exptime=1.0, out_img=sci,
                                         out_wht=wht,
                                         total_exptime=float(i))
        ok = wht > 1e-9
        rec(row="dither_inv_geometry", kernel=kernel, n=4,
            sci_max=float(np.max(sci[ok])) if np.any(ok) else None)

    # ---------------- 8. preview PNG witness (pathological cell) ----------
    sci_p, wht_p, _, _ = deposit(edge, rotated_tf(), 3, kernel="lanczos3",
                                 exptime=1.0)
    try:
        from PIL import Image
        from seestar.core.image_processing import stretch_display_data
        disp01, sp = stretch_display_data(
            np.asarray(sci_p, dtype=np.float32),
            enhanced_stretch=False, primary=True)
        disp = (np.clip(disp01, 0.0, 1.0) * 255.0).astype(np.uint8)
        img = Image.fromarray(disp)
        png = OUT / "preview_pathological_lanczos3_rot.png"
        img.save(str(png))
        MAN[str(png)] = sha(png)
        arr = np.asarray(disp, dtype=np.uint8)
        rec(row="preview_witness", png=str(png), stretch_params=sp,
            display_mean=float(arr.mean()), display_std=float(arr.std()),
            display_p99=float(np.percentile(arr, 99)))
        print("PNG witness saved:", png, "sha", MAN[str(png)][:12])
    except Exception as e:  # pragma: no cover
        import traceback
        rec(row="preview_witness", error=str(e),
            tb=traceback.format_exc(limit=3))

    # ---------------- 12. boundary detector (observables only) ------------
    def detector(sci, wht, sup1=None):
        ok = np.isfinite(sci) & (wht > 1e-9)
        s = sci[ok]
        if s.size == 0:
            return {"flagged": 0, "frac": 0.0}
        bg = float(np.median(s))
        mad = float(np.median(np.abs(s - bg))) or 1e-30
        robust_scale = 1.4826 * mad
        extreme = np.abs(s - bg) > 50.0 * robust_scale
        sup_ok = np.ones(s.shape, dtype=bool)
        if sup1 is not None:
            sup_ok = sup1[ok] > 1e-9
        flag = extreme & sup_ok
        return {"flagged": int(np.sum(flag)), "frac": float(np.mean(flag)),
                "bg": bg, "scale": robust_scale}

    for key, sci, wht, kernel, tag in cells:
        ok = detector(sci, wht)
        rec(row="detector", key=key, tag=tag, **ok)
    # false positives: square/benign lanczos full-coverage star cells
    # (included above via cells list)

    # ---------------- outputs + manifest ----------------
    with open(OUT / "rows.json", "w") as fh:
        json.dump(ROWS, fh, indent=1)
    import csv
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
        json.dump({"command": cmd, "files": MAN}, fh, indent=1)
    print(f"r2 done: {len(ROWS)} rows -> {OUT}")


if __name__ == "__main__":
    main()
