"""RF-2 — bounded M16 real-data witness (diversity + observable-noise limits).

Bounded, complementary, production-external.  This script re-uses RF-1's
faithful production-preparation basis (``research/registration_field_rotation/
m16_scale_witness.py``, a CPU-path reimplementation A/B-verified against the
production helpers in RF-1) and adds the RF-2-specific reading:

* the **rotation / translation / scale spread** across the session (diversity
  limitation — one session, one focal state, ~23 min), and
* the **held-out fit residual** (the *target-fit* residual) as the *only*
  in-run observable.

It explicitly does **not** claim M16 provides ground truth or sufficient
scale/rotation diversity (both are RF-2 non-goals).  The lifecycle POC (§ of
the report) already shows the target-fit residual is an unreliable proxy for
true-global error under a drifting target; this witness only establishes the
*level* of that observable and the *spread* of the real geometry on this one
session.  The *target-policy comparison* (immutable selected reference vs
evolving target) on these same pixels is done separately by
``m16_target_policy_witness.py``; this script is the diversity/noise
complement.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_PARENT = os.path.join(
    os.path.dirname(__file__), "..", "registration_field_rotation"
)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import m16_scale_witness as w  # noqa: E402

M16_FOLDER = "/home/tristan/M16/quick"


def run(folder=M16_FOLDER, seed=0):
    """Run RF-1's witness and aggregate the RF-2 reading.

    Returns a dict with the per-frame transforms and the aggregate spread and
    held-out residual statistics.  Raises ``FileNotFoundError`` when the data is
    absent (callers should skip).
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"M16 data not present at {folder}")

    data = w.run(folder, seed=seed)
    rows = data["rows"]
    ok = [r for r in rows if r["status"] == "ok"]
    fail = [r for r in rows if r["status"] != "ok"]

    scales = np.array([r["astroalign_scale"] for r in ok], dtype=np.float64)
    rots = np.array([r["rotation_deg"] for r in ok], dtype=np.float64)
    txs = np.array([r["tx"] for r in ok], dtype=np.float64)
    tys = np.array([r["ty"] for r in ok], dtype=np.float64)
    hold_rms = np.array([r["euclidean"]["hold_rms"] for r in ok], dtype=np.float64)
    hold_p95 = np.array([r["euclidean"]["hold_p95"] for r in ok], dtype=np.float64)

    return {
        "folder": folder,
        "n_frames": data["n_frames"],
        "reference": data["reference"],
        "n_ok": len(ok),
        "n_fail": len(fail),
        "scale_span_ppm": float((scales.max() - scales.min()) * 1e6),
        "scale_median_abs_ppm": float(np.median(np.abs(scales - 1.0)) * 1e6),
        "rot_span_deg": float(rots.max() - rots.min()),
        "rot_median_abs_deg": float(np.median(np.abs(rots))),
        "tx_span_px": float(txs.max() - txs.min()),
        "ty_span_px": float(tys.max() - tys.min()),
        "hold_rms_median_px": float(np.median(hold_rms)),
        "hold_rms_max_px": float(hold_rms.max()),
        "hold_p95_median_px": float(np.median(hold_p95)),
    }


def report(r):
    lines = []
    lines.append("# M16 real-data witness (bounded) — diversity + observable-noise limits\n")
    lines.append(
        f"folder={r['folder']}  frames={r['n_frames']}  aligned OK={r['n_ok']}  failed={r['n_fail']}  "
        f"reference={r['reference']}"
    )
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append(f"| scale range (ppm) | {r['scale_span_ppm']:.1f} |")
    lines.append(f"| median \\|scale-1\\| (ppm) | {r['scale_median_abs_ppm']:.1f} |")
    lines.append(f"| rotation span (deg) | {r['rot_span_deg']:.4f} |")
    lines.append(f"| median \\|rotation\\| (deg) | {r['rot_median_abs_deg']:.4f} |")
    lines.append(f"| translation x span (px) | {r['tx_span_px']:.1f} |")
    lines.append(f"| translation y span (px) | {r['ty_span_px']:.1f} |")
    lines.append(f"| held-out (target-fit) residual median / max (px) | {r['hold_rms_median_px']:.3f} / {r['hold_rms_max_px']:.3f} |")
    lines.append(f"| held-out P95 median (px) | {r['hold_p95_median_px']:.3f} |")
    lines.append("")
    lines.append(
        "**Limitations (explicit):** M16 is a *single* session (2025-05-30, ~23 min, "
        "one focal state) and provides **no ground truth** and **insufficient "
        "scale/rotation diversity** to resolve the cross-session/temperature "
        "scale question or to validate a target strategy.  The held-out residual "
        "above is the *target-fit* residual — the lifecycle POC shows it is an "
        "unreliable proxy for true-global error under a drifting target.  This "
        "witness therefore only bounds the *observable* noise level and the "
        "*geometry spread* on this one session.  The target-policy comparison is "
        "in ``m16_target_policy_witness.py``."
    )
    return "\n".join(lines)


def main():
    try:
        r = run()
    except FileNotFoundError as e:
        print(f"# M16 witness skipped: {e}")
        return
    print(report(r))


if __name__ == "__main__":
    main()
