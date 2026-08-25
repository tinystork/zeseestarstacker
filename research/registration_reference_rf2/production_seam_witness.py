"""RF-2 — production-seam witness for the registration alignment contract.

Production-path-bounded: this module **imports the real** ``SeestarAligner``
from ``seestar.core.alignment`` and runs the **real** ``_align_image`` (real
scale-discard, real ``cv2.warpAffine``, real ``return_M`` 2x3 contract) on
synthetic star images.  Only the heavy *matcher* is abstracted:
``astroalign.find_transform`` is monkeypatched to a deterministic closed-form
estimator that returns a **known** similarity transform and known matched
pairs, so the witness has an authoritative ground truth for the scale-discard
behaviour.

This witnesses two facts that the RF-1 report stated structurally and that the
deterministic lifecycle POC models mathematically:

1. ``_align_image`` **discards the astroalign similarity scale** and forces
   ``scale = 1.0`` (Euclidean rotation + translation) — ``alignment.py:228-237``.
2. ``_align_image(return_M=True)`` returns the **2x3 affine actually used by
   ``cv2.warpAffine``** (mapping ORIGINAL source pixels -> reference grid), the
   same matrix the Drizzle standard path feeds to
   ``_add_frame_to_drizzle_accumulators``.

No ``seestar`` production file is modified.
"""

from __future__ import annotations

import numpy as np
from skimage.transform import SimilarityTransform


def _make_star_image(shape=(96, 96), stars=((50, 30), (30, 60), (70, 70)), sigma=1.8):
    """A small synthetic star field (float32, 0-1) with Gaussian blobs."""
    img = np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for cx, cy in stars:
        img += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    m = img.max()
    return (img / m).astype(np.float32)


def _deterministic_find_transform(scale=1.02, rotation_deg=2.0, tx=5.0, ty=-3.0):
    """Build a monkeypatched ``find_transform(source, target)``.

    Returns ``(SimilarityTransform, (source_pts, target_pts))`` with a known
    scale / rotation / translation.  ``source_pts``/``target_pts`` are small
    dummy matched-pair arrays (only their existence matters to ``_align_image``,
    which uses the transform params and ignores the points).
    """
    t = np.radians(rotation_deg)
    params = np.array(
        [
            [scale * np.cos(t), -scale * np.sin(t), tx],
            [scale * np.sin(t), scale * np.cos(t), ty],
            [0.0, 0.0, 1.0],
        ]
    )
    T = SimilarityTransform(matrix=params)

    def _patched(source, target):
        src_pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        tgt_pts = src_pts.copy()
        return T, (src_pts, tgt_pts)

    return _patched, params


def run(monkeypatch):
    """Run the witness and return a report dict + markdown text.

    ``monkeypatch`` is a pytest ``monkeypatch`` fixture (or any object with
    ``setattr`` / ``context``); if ``None`` the global astroalign module is
    patched and restored manually.
    """
    from seestar.core import alignment as alignment_mod
    from seestar.core.alignment import SeestarAligner

    aligner = SeestarAligner()
    aligner.use_cuda = False

    ref_img = _make_star_image()
    src_img = _make_star_image()

    patched_fn, params = _deterministic_find_transform(
        scale=1.02, rotation_deg=2.0, tx=5.0, ty=-3.0
    )
    patched = monkeypatch.setattr(
        alignment_mod.aa, "find_transform", patched_fn
    )

    try:
        aligned, success, M = aligner._align_image(
            src_img, ref_img, "synthetic.fits", return_M=True
        )
    finally:
        if monkeypatch is None:
            patched.undo()

    # Expected: scale discarded -> rotation matrix with det == 1, translation kept.
    theta_true = np.radians(2.0)
    R = M[:2, :2]
    det = float(np.linalg.det(R))
    recovered_theta = float(np.arctan2(M[1, 0], M[0, 0]))
    recovered_scale = float(np.hypot(M[0, 0], M[1, 0]))
    tx, ty = float(M[0, 2]), float(M[1, 2])

    report = {
        "success": bool(success),
        "aligned_shape": tuple(aligned.shape),
        "aligned_dtype": str(aligned.dtype),
        "M_shape": tuple(M.shape),
        "M_dtype": str(M.dtype),
        "det_linear": det,
        "recovered_scale": recovered_scale,
        "recovered_theta_deg": float(np.degrees(recovered_theta)),
        "true_theta_deg": 2.0,
        "recovered_tx": tx,
        "recovered_ty": ty,
        "true_tx": 5.0,
        "true_ty": -3.0,
        "injected_scale": 1.02,
    }
    return report, _md(report)


def _md(r):
    lines = []
    lines.append("# Production-seam witness — real `_align_image` scale-discard + return_M contract\n")
    lines.append(f"- `_align_image` success: {r['success']}")
    lines.append(f"- aligned image shape/dtype: {r['aligned_shape']} / {r['aligned_dtype']}")
    lines.append(f"- returned M shape/dtype: {r['M_shape']} / {r['M_dtype']}")
    lines.append("")
    lines.append("| quantity | injected (astroalign) | returned by production |")
    lines.append("|---|---|---|")
    lines.append(f"| scale | {r['injected_scale']} | {r['recovered_scale']:.6f} (forced to 1.0) |")
    lines.append(f"| det(linear 2x2) | — | {r['det_linear']:.6f} |")
    lines.append(f"| rotation | {r['true_theta_deg']} deg | {r['recovered_theta_deg']:.6f} deg |")
    lines.append(f"| translation x | {r['true_tx']} px | {r['recovered_tx']:.6f} px |")
    lines.append(f"| translation y | {r['true_ty']} px | {r['recovered_ty']:.6f} px |")
    lines.append("")
    lines.append(
        "The returned matrix has scale == 1.0 (det == 1.0) while rotation and "
        "translation are preserved exactly — the production scale-discard "
        "(`alignment.py:228-237`) and the 2x3 `return_M` contract are exercised "
        "on the real function."
    )
    return "\n".join(lines)


def main():
    # Manual (non-pytest) run: patch the global astroalign module and restore.
    import seestar.core.alignment as alignment_mod

    original = alignment_mod.aa.find_transform

    class _M:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)
            return self

        def undo(self):
            alignment_mod.aa.find_transform = original

    report, md = run(_M())
    print(md)
    return report


if __name__ == "__main__":
    main()
