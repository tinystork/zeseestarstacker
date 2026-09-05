"""Autonomous GPU capability probe and authoritative acceleration policy.

This module gives ZeSeestarStacker its own GPU detection layer, independent of
the legacy ``check_cuda`` / ``check_cupy_cuda`` helpers and of ZeAlfie.  It is
the single source of truth for *whether* the production GPU backend (CuPy)
exists on this machine and *which* backend the stacker should use for a run.

Design rules:

* ``probe_gpu()`` is autonomous, non-destructive, reasonably fast and safe to
  call from anywhere (GUI, CLI, headless runs, tests).  It never raises and
  never imports optional libraries at module import time: CuPy and the
  ``cv2.cuda`` submodule are only touched defensively at probe time.
* CuPy is the SOLE production GPU backend.  OpenCV CUDA (``cv2.cuda``) is
  reported by the probe but is **diagnostic-only**: no production operation
  uses it, and it never makes ``backend_ready``/``state`` resolve to ready.
* Five canonical, machine-readable states are distinguished:

  * ``no_gpu``            -- no supported GPU detected anywhere.
  * ``gpu_no_runtime``    -- GPU present but the CUDA runtime cannot reach it.
  * ``cuda_no_backend``   -- CUDA/driver present but no usable Python backend
                             (including OpenCV-CUDA-only machines: CuPy is
                             unavailable while OpenCV CUDA is present but
                             diagnostic-only).
  * ``backend_error``     -- the CuPy backend found the device but failed to
                             initialize (e.g. real-kernel JIT failure caused
                             by missing CUDA toolkit headers).
  * ``ready``             -- the CuPy production backend actually works.

* ``AccelerationPolicy`` resolves the backend exactly once per run and is the
  authoritative decision record handed to the rest of the application.

Nothing in this module wires into the stacker, GUI or run contract; those are
later milestones.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

__all__ = [
    "STATE_NO_GPU",
    "STATE_GPU_NO_RUNTIME",
    "STATE_CUDA_NO_BACKEND",
    "STATE_BACKEND_ERROR",
    "STATE_READY",
    "GpuCapabilities",
    "AccelerationPolicy",
    "probe_gpu",
]

# Canonical capability states (machine-readable).
STATE_NO_GPU = "no_gpu"  # no supported GPU detected
STATE_GPU_NO_RUNTIME = "gpu_no_runtime"  # GPU present but CUDA runtime unavailable
STATE_CUDA_NO_BACKEND = "cuda_no_backend"  # CUDA available but required Python backend unavailable
STATE_BACKEND_ERROR = "backend_error"  # GPU backend initialization failure
STATE_READY = "ready"  # the CuPy production backend initialized


def _truncate(text: object, limit: int = 200) -> str:
    """Shorten an exception message for storage in ``failure_reason``."""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


@dataclass(frozen=True)
class GpuCapabilities:
    """Snapshot of the GPU situation at probe time (immutable)."""

    gpu_detected: bool  # a supported GPU is physically present
    cuda_runtime_ready: bool  # CUDA runtime/driver usable
    cupy_ready: bool  # CuPy importable AND a real kernel initializes
    # Diagnostic only: no production operation uses OpenCV CUDA.  Reported for
    # information, but it must NEVER drive backend_ready / state==ready.
    opencv_cuda_ready: bool  # cv2.cuda has >= 1 enabled device (diagnostic only)
    # A production GPU backend is ready == cupy_ready only (CuPy is the sole
    # production backend; OpenCV CUDA is diagnostic-only).
    backend_ready: bool  # cupy_ready
    device_name: str | None  # e.g. "NVIDIA GeForce MX150"
    device_vram_mb: int | None
    compute_capability: str | None  # e.g. "6.1"
    failure_reason: str | None  # why not ready / which backend failed, else None
    state: str  # one of the STATE_* constants above

    def describe(self) -> str:
        """Human-readable single line for the GUI."""
        if self.state == STATE_READY:
            identity = self.device_name or "CUDA GPU"
            return f"{identity} — CUDA ready (CuPy)"
        if self.state == STATE_NO_GPU:
            return "No compatible GPU detected"
        if self.state == STATE_GPU_NO_RUNTIME:
            return "GPU present but CUDA runtime unavailable"
        reason = f" ({self.failure_reason})" if self.failure_reason else ""
        return f"GPU present but backend unavailable{reason}"


def _query_nvidia_smi() -> tuple[str, int] | None:
    """Best-effort hardware presence query; never raises.

    Returns ``(name, vram_mb)`` for the first GPU, else ``None``.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        name, _, memory = line.rpartition(",")
        name = name.strip()
        if not name:
            continue
        try:
            vram_mb = int(memory.strip())
        except ValueError:
            vram_mb = None
        return name, vram_mb
    return None


def probe_gpu() -> GpuCapabilities:
    """Autonomous, non-destructive GPU capability probe.

    Never raises: every optional dependency and subprocess call is wrapped.
    Distinguishes the five ``STATE_*`` failure/success states deterministically.
    """
    cupy_failure: str | None = None
    opencv_failure: str | None = None
    cp_import_ok = False
    cp_device_ok = False
    cp_kernel_ok = False
    ocv_device = False
    device_name: str | None = None
    device_vram_mb: int | None = None
    compute_capability: str | None = None

    # --- (a) OpenCV CUDA: cv2.cuda.getCudaEnabledDeviceCount() > 0 --------
    cv2 = None  # noqa: F841 - local alias, may stay None on import failure
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - broken environment
        opencv_failure = f"opencv: import failed ({type(exc).__name__}: {_truncate(exc)})"
    if cv2 is not None:
        try:
            cuda_mod = getattr(cv2, "cuda", None)
            if cuda_mod is not None:
                count = int(cuda_mod.getCudaEnabledDeviceCount())
                ocv_device = count > 0
        except Exception as exc:
            opencv_failure = (
                f"opencv-cuda: {type(exc).__name__}: {_truncate(exc)}"
            )

    # --- (b) CuPy: importable AND a real kernel actually initializes -------
    cp = None  # noqa: F841
    try:
        import cupy as cp
        cp_import_ok = True
    except Exception as exc:
        cupy_failure = (
            f"cupy: import failed ({type(exc).__name__}: {_truncate(exc)})"
        )
    if cp_import_ok:
        try:
            cp_device_ok = bool(cp.cuda.is_available())
        except Exception as exc:
            cupy_failure = (
                f"cupy: availability check failed "
                f"({type(exc).__name__}: {_truncate(exc)})"
            )
            cp_device_ok = False
        if cp_device_ok:
            # CuPy may report is_available()==True yet fail on a real kernel
            # when CUDA toolkit headers are missing (JIT cannot compile).
            try:
                probe_arr = cp.arange(4, dtype=cp.float32)
                doubled = probe_arr * 2
                host_values = [float(value) for value in doubled.get()]
                if host_values != [0.0, 2.0, 4.0, 6.0]:
                    raise RuntimeError(
                        f"unexpected elementwise kernel result {host_values!r}"
                    )
                cp_kernel_ok = True
            except Exception as exc:
                cupy_failure = (
                    f"cupy: real-kernel init failed "
                    f"({type(exc).__name__}: {_truncate(exc)})"
                )
            # Device identity is best-effort and does not require JIT.
            try:
                device_id = cp.cuda.runtime.getDevice()
                props = cp.cuda.runtime.getDeviceProperties(device_id)
                raw_name = props.get("name")
                if isinstance(raw_name, bytes):
                    raw_name = raw_name.decode("utf-8", "replace")
                raw_name = str(raw_name).strip()
                if raw_name:
                    device_name = raw_name
                major = int(props.get("major", 0) or 0)
                minor = int(props.get("minor", 0) or 0)
                if major or minor:
                    compute_capability = f"{major}.{minor}"
                total_bytes = int(props.get("totalGlobalMem", 0) or 0)
                if total_bytes > 0:
                    device_vram_mb = total_bytes // (1024 * 1024)
            except Exception:
                pass  # identity is informational only

    # --- (c) Hardware presence fallback: nvidia-smi -------------------------
    smi_gpu = False
    if not (cp_device_ok or ocv_device):
        try:
            smi_result = _query_nvidia_smi()
        except Exception:
            smi_result = None
        if smi_result is not None:
            smi_gpu = True
            smi_name, smi_vram_mb = smi_result
            if device_name is None and smi_name:
                device_name = smi_name
            if device_vram_mb is None and smi_vram_mb:
                device_vram_mb = smi_vram_mb

    # --- (d) Deterministic state resolution ---------------------------------
    # Hardware presence / runtime evidence may count OpenCV CUDA, but the
    # production backend is CuPy only: backend_ready and state==ready are
    # driven exclusively by cp_kernel_ok (F1).
    gpu_detected = bool(cp_device_ok or ocv_device or smi_gpu)
    cuda_runtime_ready = bool(cp_device_ok or ocv_device)
    backend_ready = bool(cp_kernel_ok)

    if cp_kernel_ok:
        # The CuPy production backend actually works.
        state = STATE_READY
        failure_reason = None
    elif not gpu_detected:
        state = STATE_NO_GPU
        failure_reason = None
    elif cp_device_ok and not cp_kernel_ok:
        # Device visible to CuPy but the real kernel failed to initialize.
        state = STATE_BACKEND_ERROR
        failure_reason = cupy_failure
    elif cp_import_ok and not cp_device_ok and not ocv_device:
        # GPU visible (nvidia-smi) but the CUDA runtime cannot reach it.
        state = STATE_GPU_NO_RUNTIME
        failure_reason = cupy_failure or (
            "cupy: CUDA runtime cannot reach the GPU"
        )
    else:
        # CUDA-capable hardware present, but the CuPy production backend is
        # unavailable (absent, runtime-unreachable, or kernel-broken).  When
        # OpenCV CUDA is present it is reported but unused (diagnostic only).
        state = STATE_CUDA_NO_BACKEND
        parts = [part for part in (cupy_failure, opencv_failure) if part]
        if ocv_device:
            parts.append(
                "OpenCV-CUDA is present but diagnostic-only: CuPy is the sole "
                "production backend and is unavailable"
            )
        failure_reason = (
            "; ".join(parts)
            if parts
            else "no CUDA-capable Python backend available (CuPy is the sole "
                 "production backend)"
        )

    return GpuCapabilities(
        gpu_detected=gpu_detected,
        cuda_runtime_ready=cuda_runtime_ready,
        cupy_ready=cp_kernel_ok,
        opencv_cuda_ready=ocv_device,
        backend_ready=backend_ready,
        device_name=device_name,
        device_vram_mb=device_vram_mb,
        compute_capability=compute_capability,
        failure_reason=failure_reason,
        state=state,
    )


@dataclass(frozen=True)
class AccelerationPolicy:
    """Resolves backend selection ONCE per run. Authoritative single source.

    Resolvable backends are ``"cpu"`` and ``"cupy"`` only: CuPy is the sole
    production GPU backend (``opencv_cuda`` is diagnostic-only and never a
    resolvable backend).
    """

    capabilities: GpuCapabilities
    request_gpu: bool = False

    @property
    def backend(self) -> str:
        """Resolved backend for this run: "cpu" | "cupy" (never opencv_cuda)."""
        if not self.request_gpu:
            return "cpu"
        if self.capabilities.cupy_ready:
            return "cupy"
        return "cpu"

    @property
    def fallback_reason(self) -> str | None:
        """Why we run CPU although ``request_gpu`` is True, else None."""
        if self.backend != "cpu" or not self.request_gpu:
            return None
        return self.capabilities.failure_reason or self.capabilities.state

    def describe(self) -> str:
        """GUI-friendly single line describing the resolved state."""
        capabilities = self.capabilities
        if self.backend == "cpu":
            if not self.request_gpu:
                return "CPU stacking (GPU acceleration not requested)"
            return f"CPU stacking (GPU requested but unavailable: {self.fallback_reason})"
        identity = capabilities.device_name or "CUDA GPU"
        return f"CuPy acceleration on {identity}"
