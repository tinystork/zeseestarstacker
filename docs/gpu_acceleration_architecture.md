# GPU Acceleration Architecture (ZeSeestarStacker)

Status: reflects the final architecture after the GPU-modernization pass
(probe + policy in `seestar/core/gpu.py`; CuPy-only backend in
`seestar/core/stack_gpu.py`; GUI status in the Qt System tab; Boring
subprocess seam).

## Design summary

ZSSS has **one** acceleration policy. CPU is the complete, canonical reference
path; GPU acceleration is strictly optional and adds CuPy execution only for a
well-defined subset of the stacking reductions.

```
GpuCapabilities (probe_gpu: 5-state machine)
        │
        ▼
AccelerationPolicy (resolved once, frozen per run)
        │  request_gpu (user intent) × capabilities
        ▼
effective_backend  →  "cpu" | "cupy"   (never "opencv_cuda")
        │
        ▼
_stack_batch reductions routed through _gpu_reduce:
        kappa-sigma / linear-fit-clip / median → GPU when effective_backend
        == "cupy" AND the stack fits in free VRAM (dynamic guard);
        otherwise CPU.  Any GPU failure → logged warning + CPU fallback.
```

## Components

* **`seestar/core/gpu.py`** — `probe_gpu()` returns an immutable
  `GpuCapabilities` snapshot with one of five states:

  | state | meaning |
  |---|---|
  | `no_gpu` | no supported GPU detected anywhere |
  | `gpu_no_runtime` | GPU present but the CUDA runtime cannot reach it |
  | `cuda_no_backend` | CUDA-capable hardware present but the CuPy production backend unavailable (includes OpenCV-CUDA-only machines) |
  | `backend_error` | CuPy sees the device but a real kernel failed to initialize |
  | `ready` | the CuPy production backend actually works |

  `backend_ready` means `cupy_ready` **only**. `opencv_cuda_ready` is
  diagnostic-only: no production operation uses OpenCV CUDA, and it never
  drives `backend_ready` / `state` / `AccelerationPolicy.backend`.

  `AccelerationPolicy(capabilities, request_gpu)` resolves the backend exactly
  once per run: `"cpu"` or `"cupy"`. It is frozen at the run boundary
  (`SeestarQueuedStacker.start_processing`), so later mutation of
  `request_gpu` cannot change the backend of an ongoing run.

* **`seestar/core/stack_gpu.py`** — CuPy kernels that are exact twins of the
  CPU reference algorithms in `seestar/core/stack_methods.py` (untouched) for
  the **sorting-based** reductions: kappa-sigma, linear-fit-clip, median.
  NaN == missing sample; identical masks, floors, `_rejected_pct` formula and
  weight maps; results are always returned as NumPy arrays.

* **What stays CPU**: winsorized-sigma-clip (the default rejection), mean,
  alignment, drizzle, reprojection, feathering, quality metrics, streaming and
  tiled paths. CuPy accelerates *only* kappa-sigma / linear-fit-clip / median
  when selected and VRAM-fitting.

## VRAM-dynamic eligibility

`SeestarQueuedStacker._reduction_xp` evaluates eligibility **dynamically** for
every call against the device's *actual current free memory*
(`cuda.runtime.memGetInfo`): an estimated footprint (stack + sorting peak,
~4× the float32 stack) must fit in 60 % of free VRAM, else the reduction runs
on CPU. There is **no fixed workload-size (stack-count) threshold**.

**Hardware-specific measurement (MX150, do not generalize)**: the measured
CPU/GPU crossover points below were obtained on an NVIDIA GeForce MX150
(2 GB, compute capability 6.1) with CuPy 14.2.0, 1080×1920 float32 stacks,
and are NOT a universal claim:

* N=20 kappa-sigma: CPU ≈ 1.92 s, GPU ≈ 0.99 s (≈ 1.9× faster).
* N=30 kappa-sigma: CPU ≈ 4.01 s, GPU ≈ 1.46 s (≈ 2.7× faster).
* N=50 kappa-sigma: CPU ≈ 13.5 s; GPU does NOT fit in 2 GB (real OOM
  measured) → the dynamic VRAM guard routes it to CPU.

A larger/faster GPU admits larger workloads automatically through the same
guard; transfer/execution-cost tuning for tiny stacks is deliberately NOT
encoded as a fixed threshold so that larger GPUs are never excluded.

## Runtime behaviour

* User intent (the GUI "Use GPU" toggle) is stored as
  `QtSettingsState.use_gpu`, crosses the seam as the field key `"use_gpu"`,
  and is applied to the stacker instance as `request_gpu` by the backend
  adapter; the stacker resolves its own probe + `AccelerationPolicy` (no
  hardware assumption travels from the GUI).
* The Qt GUI probes capability off the main thread (background worker) and
  shows the resolved state, e.g. `GPU acceleration enabled — CuPy / NVIDIA
  GeForce MX150`, or `No compatible GPU detected`.
* If a GPU reduction fails at runtime, the queue manager logs a warning and
  automatically reruns that reduction on CPU.

## Boring (single-batch CSV) subprocess seam

The Qt Boring Stack route launches `seestar/gui/boring_stack.py` as a separate
process. Only the boolean intent crosses the boundary (`--gpu` / `--no-gpu`);
the subprocess constructs `SeestarQueuedStacker(gpu=args.request_gpu)` and
resolves the same probe/policy inside the subprocess. Boring's default
reduction is `stacking_mode="winsorized-sigma"`, which remains CPU-only, so
the default Boring run still executes on CPU even with `--gpu`; the intent is
in place for eligible reductions (kappa-sigma / linear-fit-clip / median)
should a Boring mode select them.

## ZeAlfie follow-up (separate repo, NOT performed here)

ZeAlfie is a separate repository and was **not** modified in this pass. The
follow-up required there is documentation-only:

* (a) The acceleration dependency closure in ZeAlfie's `products.toml`
  remains UNCHANGED and still correct: `cupy-cuda12x` plus the
  `nvidia-*-cu12` 12.4.x closure.
* (b) Only the stale DESCRIPTION should be updated — from *"CuPy OPTIONALLY
  for its drizzle step (`drizzle_utils.py` / `check_cupy_cuda`)"* to *"CuPy
  OPTIONALLY for sorting-based stacking reductions
  (`seestar/core/stack_gpu.py`; probe `seestar/core/gpu.py`)"*.
