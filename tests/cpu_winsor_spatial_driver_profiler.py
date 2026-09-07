#!/usr/bin/env python
"""Isolated RSS witness for the real exact-N CPU spatial Winsor driver.

Each row builds already-resident observations, records process RSS, executes
``stack_winsorized_sigma_cpu_tiled`` (or the canonical FULL reference), and
samples native process RSS with psutil.  Spatial rows keep tile geometry fixed
while full-frame geometry changes, exposing any accidental second full-frame
N-cube allocation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SCENE = r'''
import gc, json, os, resource, sys, threading, time
import numpy as np
import psutil
from seestar.core.cpu_winsor_exact_n import (
    cpu_tile_demand_bytes, stack_winsorized_sigma_cpu_tiled,
)
from seestar.core.stack_methods import _stack_winsorized_sigma_iter

spec = json.loads(sys.argv[1])
rng = np.random.default_rng(840)
N, H, W, C = spec["N"], spec["H"], spec["W"], spec["C"]
shape = (H, W) if C == 1 else (H, W, C)
images = []
for i in range(N):
    image = rng.normal(100.0, 5.0, size=shape).astype(np.float32)
    image.reshape(-1)[i % image.size] += 500.0
    images.append(image)
gc.collect()
process = psutil.Process(os.getpid())
baseline = process.memory_info().rss
peak = {"rss": baseline}
stop = threading.Event()
def watch():
    while not stop.is_set():
        peak["rss"] = max(peak["rss"], process.memory_info().rss)
        time.sleep(0.0005)
thread = threading.Thread(target=watch, daemon=True)
thread.start()
t0 = time.perf_counter()
if spec["mode"] == "spatial":
    tile = tuple(spec["tile"])
    out = stack_winsorized_sigma_cpu_tiled(
        images, None, tile_shape=tile, winsor_limits=(0.05, 0.05),
        max_iters=5, return_weights=True,
    )
else:
    tile = (H, W)
    out = _stack_winsorized_sigma_iter(
        images, None, winsor_limits=(0.05, 0.05), max_iters=5,
        max_mem_bytes=1 << 62, return_weights=True,
    )
elapsed = time.perf_counter() - t0
stop.set(); thread.join(timeout=2)
peak["rss"] = max(peak["rss"], process.memory_info().rss)
tile_demand, factor, scratch = cpu_tile_demand_bytes(
    N, tile[0], tile[1], C, 4, (0.05, 0.05)
)
full_cube = N * H * W * C * 4
output_bytes = 2 * H * W * C * 4
print(json.dumps({
    **spec,
    "baseline_rss_bytes": baseline,
    "peak_rss_bytes": peak["rss"],
    "rss_delta_bytes": peak["rss"] - baseline,
    "resource_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    "resident_input_bytes": full_cube,
    "forbidden_second_full_cube_bytes": full_cube,
    "planner_tile_demand_bytes": tile_demand,
    "output_sci_wht_bytes": output_bytes,
    "planner_plus_outputs_bytes": tile_demand + output_bytes,
    "factor": factor,
    "scratch_bytes": scratch,
    "elapsed_s": elapsed,
    "output_shape": list(out[0].shape),
    "scientific_n": N,
}))
'''

ROWS = [
    {"name": "full_rgb_256_n20", "mode": "full", "N": 20, "H": 256, "W": 256, "C": 3},
    {"name": "spatial_rgb_256_n20_t64", "mode": "spatial", "N": 20, "H": 256, "W": 256, "C": 3, "tile": [64, 64]},
    {"name": "spatial_rgb_512_n20_t64", "mode": "spatial", "N": 20, "H": 512, "W": 512, "C": 3, "tile": [64, 64]},
    {"name": "spatial_rgb_256_n36_t64", "mode": "spatial", "N": 36, "H": 256, "W": 256, "C": 3, "tile": [64, 64]},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = []
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", SEESTAR_USE_SCIPY_WINSOR="0")
    for row in ROWS:
        proc = subprocess.run(
            [sys.executable, "-c", SCENE, json.dumps(row)], cwd=REPO,
            env=env, capture_output=True, text=True, timeout=1200,
        )
        if proc.returncode:
            raise RuntimeError(f"{row['name']} failed: {proc.stderr[-4000:]}")
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    Path(args.out).write_text(json.dumps({"rows": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
