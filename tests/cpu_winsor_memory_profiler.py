#!/usr/bin/env python
"""8.4.0 pre-W80 stage B — isolated-subprocess RSS profiling of the canonical
CPU Winsorized sigma reducer (seestar.core.stack_methods).

Mission ``zsss-840-prew80-20260907``.  Standalone evidence helper (NOT a
pytest test): each matrix row runs in its OWN subprocess so native NumPy /
SciPy / Bottleneck allocations and process peak RSS are measured cleanly
(no tracemalloc-only numbers).

Measured quantity per row
-------------------------
* baseline RSS after interpreter + imports + building the resident input
  observations (the input cube is ALREADY resident — counted once);
* process peak RSS around ``_stack_winsorized_sigma_iter``
  (``resource.getrusage(RUSAGE_SELF).ru_maxrss`` high-water mark, plus a
  psutil RSS watcher as a sanity cross-check);
* ``delta_over_input_cube_bytes = peak - baseline``: the working set the
  reducer needs ABOVE the already-resident observations — never subtracting
  the resident input twice from available RAM.

Rows cover mono/RGB x N in {10,20,36,50} x full-frame / full-width band /
rectangular tile shapes x winsor rank 0 / rank>0 x apply_rewinsor T/F x
weighted/unweighted x NumPy default backend (+ one SciPy opt-in row;
Bottleneck presence is reported, it is installed in this venv).

Usage:
  .venv/bin/python tests/cpu_winsor_memory_profiler.py \
      --outdir <evidence-dir>/stage-b-evidence [--row-row ...]
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = "/home/tristan/.openclaw/workspace/projects/zeseestarstacker"
MB = 1024 * 1024


# ---------------------------------------------------------------------------
# Deterministic synthetic scene (shared by every subprocess row)
# ---------------------------------------------------------------------------
SCENE_PY = r"""
import json, os, sys, time, resource, threading
import numpy as np
from seestar.core.stack_methods import _stack_winsorized_sigma_iter, USE_SCIPY_WINSOR, SCIPY_AVAILABLE

try:
    import bottleneck as _bn
    BN_AVAILABLE = True
except Exception:
    _bn = None
    BN_AVAILABLE = False

spec = json.loads(sys.argv[1])
rng = np.random.default_rng(spec["seed"])
H, W, C = spec["H"], spec["W"], spec["C"]
N = spec["N"]
shape2 = (H, W)
full_shape = (H, W, C) if C > 1 else (H, W)
# sky + smooth gradient + noise + sparse outliers (+ optional NaN support gaps)
yy = (np.arange(H, dtype=np.float32)[:, None] + 1.0) / float(H)
xx = (np.arange(W, dtype=np.float32)[None, :] + 1.0) / float(W)
base = 800.0 + 60.0 * yy + 40.0 * xx
base3 = None
if C > 1:
    chan = np.stack([base, base * 0.9 + 30.0, base * 1.1 - 20.0], axis=-1)
    base3 = np.broadcast_to(chan[None, ...], (N, H, W, C)).copy()
images = []
for i in range(N):
    noise = rng.standard_normal(full_shape).astype(np.float32) * 8.0
    spike = (rng.random(full_shape) < 0.01).astype(np.float32) * rng.uniform(
        200.0, 900.0, size=full_shape
    ).astype(np.float32)
    img = (base if C == 1 else base3[i]) + noise + spike
    if spec.get("nan_frac", 0.0) > 0.0:
        img = img.copy()
        m = rng.random(full_shape) < spec["nan_frac"]
        if C > 1:
            m = m | (rng.random(full_shape) < spec["nan_frac"])
        img[m] = np.nan
    images.append(img.astype(np.float32))

weights = None
if spec["weighted"]:
    weights = rng.uniform(0.5, 1.5, size=N).astype(np.float32)

input_bytes = N * H * W * C * 4

# baseline AFTER interpreter+imports+resident inputs (inputs counted once)
baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

peak_sampled = {"rss": baseline}

def _watch():
    try:
        import psutil
        p = psutil.Process(os.getpid())
        while not _stop.is_set():
            rss = p.memory_info().rss
            if rss > peak_sampled["rss"]:
                peak_sampled["rss"] = rss
            time.sleep(0.001)
    except Exception:
        pass

_stop = threading.Event()
th = threading.Thread(target=_watch, daemon=True)
th.start()
t0 = time.perf_counter()
out = _stack_winsorized_sigma_iter(
    images,
    weights,
    kappa=3.0,
    winsor_limits=tuple(spec["winsor_limits"]),
    apply_rewinsor=spec["apply_rewinsor"],
    max_iters=5,
    kappa_decay=0.9,
    max_mem_bytes=1 << 62,
    return_weights=True,
)
dt = time.perf_counter() - t0
_stop.set()
th.join(timeout=2.0)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
print(json.dumps({
    "ok": True,
    "baseline_rss": int(baseline),
    "peak_rss": int(peak),
    "peak_rss_sampled": int(peak_sampled["rss"]),
    "delta_over_input_cube_bytes": int(peak - baseline),
    "input_cube_bytes": int(input_bytes),
    "elapsed_s": round(dt, 4),
    "scipy_enabled": bool(USE_SCIPY_WINSOR and SCIPY_AVAILABLE),
    "bottleneck_available": bool(BN_AVAILABLE),
    "backend": "scipy" if (USE_SCIPY_WINSOR and SCIPY_AVAILABLE) else "numpy",
    "out_shape": list(np.shape(out[0])),
    "weighted": bool(spec["weighted"]),
}))
"""


def _backend_env(scipy_row):
    env = dict(os.environ)
    env["SEESTAR_USE_SCIPY_WINSOR"] = "1" if scipy_row else "0"
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def run_row(spec, scipy_row=False):
    """Run one isolated subprocess row and return its parsed JSON result."""
    cmd = [sys.executable, "-c", SCENE_PY, json.dumps(spec)]
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        env=_backend_env(scipy_row),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"row {spec.get('name')} failed rc={proc.returncode}: {proc.stderr[-4000:]}"
        )
    out = proc.stdout.strip().splitlines()[-1]
    return json.loads(out)


def build_matrix():
    """Deterministic row matrix (name -> kwargs) covering the required dims."""
    rows = []

    def add(name, H, W, C, N, winsor=(0.05, 0.05), rewinsor=True,
            weighted=False, nan_frac=0.0, seed=11):
        rows.append(dict(
            name=name, H=H, W=W, C=C, N=N, winsor_limits=list(winsor),
            apply_rewinsor=bool(rewinsor), weighted=bool(weighted),
            nan_frac=nan_frac, seed=seed,
        ))

    # mono full-frame, N sweep (rank>0 for N>=20 at 5%; N=10 is rank 0)
    add("mono_256_N10_rank0", 256, 256, 1, 10)
    add("mono_256_N20", 256, 256, 1, 20)
    add("mono_256_N36", 256, 256, 1, 36)
    add("mono_256_N50", 256, 256, 1, 50)
    # mono full-frame larger spatial
    add("mono_512_N20_full", 512, 512, 1, 20)
    add("mono_512_N50_full", 512, 512, 1, 50)
    # rewinsor off
    add("mono_256_N36_rwF", 256, 256, 1, 36, rewinsor=False)
    # weighted
    add("mono_256_N36_w", 256, 256, 1, 36, weighted=True)
    # rank-0 via tiny limits at large N
    add("mono_256_N50_rank0_lim0005", 256, 256, 1, 50, winsor=(0.005, 0.005))
    # NaN support gaps
    add("mono_256_N50_nan2pct", 256, 256, 1, 50, nan_frac=0.02)
    # full-width horizontal band (tile emulation) and rectangular tile
    add("band_128x512_N36", 128, 512, 1, 36)
    add("rect_128x256_N36", 128, 256, 1, 36)
    # small spatial (tiny overhead floor)
    add("mono_64_N10_small", 64, 64, 1, 10)
    add("mono_64_N50_small", 64, 64, 1, 50)
    # RGB full-frame
    add("rgb_128_N20", 128, 128, 3, 20)
    add("rgb_128_N50", 128, 128, 3, 50)
    add("rgb_256_N20_full", 256, 256, 3, 20)
    add("rgb_256_N36_full", 256, 256, 3, 36)
    add("rgb_256_N50_rwF", 256, 256, 3, 50, rewinsor=False)
    add("rgb_256_N36_w", 256, 256, 3, 36, weighted=True)
    add("rgb_256_N50_rank0_lim0005", 256, 256, 3, 50, winsor=(0.005, 0.005))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(
        Path(__file__).resolve().parents[1]
        / ".a2a-reports" / "zsss-840-prew80-20260907" / "stage-b-evidence"
    ))
    ap.add_argument("--rows", default="all",
                    help="comma list of row names, or 'all'")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = build_matrix()
    if args.rows != "all":
        keep = set(args.rows.split(","))
        rows = [r for r in rows if r["name"] in keep]

    results = []
    scipy_done = False
    for spec in rows:
        scipy_row = False
        if spec["name"].startswith("rgb_256_N20_full") and not scipy_done:
            scipy_row = True  # representative SciPy opt-in row
            scipy_done = True
        print(f"RUN {spec['name']} scipy={scipy_row} ...", flush=True)
        t0 = time.time()
        res = run_row(spec, scipy_row=scipy_row)
        res["name"] = spec["name"]
        res["shape"] = [spec["H"], spec["W"]]
        res["N"] = spec["N"]
        res["C"] = spec["C"]
        res["dtype"] = "float32"
        res["winsor_limits"] = spec["winsor_limits"]
        res["apply_rewinsor"] = spec["apply_rewinsor"]
        res["weighted"] = spec["weighted"]
        res["nan_frac"] = spec["nan_frac"]
        res["wall_s"] = round(time.time() - t0, 2)
        results.append(res)
        print(f"  ok delta={res['delta_over_input_cube_bytes']/MB:.1f} MiB "
              f"peak={res['peak_rss']/MB:.1f} MiB "
              f"baseline={res['baseline_rss']/MB:.1f} MiB "
              f"({res['elapsed_s']}s)", flush=True)

    json_path = outdir / "cpu_winsor_rss_rows.json"
    json_path.write_text(
        json.dumps({"rows": results}, indent=1), encoding="utf-8"
    )
    print(f"WROTE {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
