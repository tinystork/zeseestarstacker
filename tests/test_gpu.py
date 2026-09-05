"""Tests for the GPU capability probe and acceleration policy.

Exercises :mod:`seestar.core.gpu` deterministically through mocks so the
suite is green on any host (with or without a GPU, CuPy or OpenCV-CUDA), plus
one unmocked coherence smoke test that only asserts the probe never raises and
returns a self-consistent :class:`GpuCapabilities`.

Five states are each proven reachable:

* ``no_gpu``            -- no GPU evidence anywhere.
* ``gpu_no_runtime``    -- nvidia-smi sees a GPU, CuPy importable but its CUDA
                           runtime cannot reach the device.
* ``cuda_no_backend``   -- nvidia-smi sees a GPU, CuPy absent, cv2.cuda absent.
* ``backend_error``     -- CuPy sees the device but the real kernel fails
                           (e.g. missing CUDA toolkit headers -> JIT failure).
* ``ready``             -- CuPy kernel or OpenCV-CUDA device actually works.
"""

import sys
import types

import seestar.core.gpu as gpu_module
from seestar.core.gpu import (
    STATE_BACKEND_ERROR,
    STATE_CUDA_NO_BACKEND,
    STATE_GPU_NO_RUNTIME,
    STATE_NO_GPU,
    STATE_READY,
    AccelerationPolicy,
    GpuCapabilities,
    probe_gpu,
)

ALL_STATES = {
    STATE_NO_GPU,
    STATE_GPU_NO_RUNTIME,
    STATE_CUDA_NO_BACKEND,
    STATE_BACKEND_ERROR,
    STATE_READY,
}


# ---------------------------------------------------------------------------
# mock helpers
# ---------------------------------------------------------------------------


class _FakeArray:
    """Minimal CuPy device-array stand-in (elementwise ``*`` + ``.get()``)."""

    def __init__(self, values):
        self._values = list(values)

    def __mul__(self, other):
        return _FakeArray([value * other for value in self._values])

    def get(self):
        return self._values


def _fake_cupy_module(*, available=True, kernel_error=None,
                      device_name=b"Fake NVIDIA GPU", major=8, minor=0,
                      vram_bytes=8 * 1024 ** 3):
    """Build a fake ``cupy`` module with tunable behavior."""

    def arange(count, dtype=None):
        if kernel_error is not None:
            raise kernel_error
        return _FakeArray(float(i) for i in range(count))

    runtime = types.SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda dev: {
            "name": device_name,
            "major": major,
            "minor": minor,
            "totalGlobalMem": vram_bytes,
        },
    )
    cuda = types.SimpleNamespace(is_available=lambda: available, runtime=runtime)
    return types.SimpleNamespace(arange=arange, cuda=cuda, float32="float32")


def _fake_cv2_module(*, cuda_devices=None):
    """Fake ``cv2``: no ``.cuda`` by default, or a fixed device count."""
    if cuda_devices is None:
        return types.SimpleNamespace()
    cuda = types.SimpleNamespace(
        getCudaEnabledDeviceCount=lambda: cuda_devices
    )
    return types.SimpleNamespace(cuda=cuda)


def _no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(gpu_module, "_query_nvidia_smi", lambda: None)


def _caps(**overrides):
    """Build a :class:`GpuCapabilities` with sane defaults."""
    base = dict(
        gpu_detected=False,
        cuda_runtime_ready=False,
        cupy_ready=False,
        opencv_cuda_ready=False,
        backend_ready=False,
        device_name=None,
        device_vram_mb=None,
        compute_capability=None,
        failure_reason=None,
        state=STATE_NO_GPU,
    )
    base.update(overrides)
    return GpuCapabilities(**base)


# ---------------------------------------------------------------------------
# probe: five reachable states
# ---------------------------------------------------------------------------


def test_probe_cpu_only_reports_no_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_module())  # no .cuda
    monkeypatch.setitem(sys.modules, "cupy", None)  # import -> ImportError
    _no_nvidia_smi(monkeypatch)

    caps = probe_gpu()

    assert caps.state == STATE_NO_GPU
    assert caps.state in {STATE_NO_GPU, STATE_GPU_NO_RUNTIME, STATE_CUDA_NO_BACKEND}
    assert caps.gpu_detected is False
    assert caps.cuda_runtime_ready is False
    assert caps.cupy_ready is False
    assert caps.opencv_cuda_ready is False
    assert caps.backend_ready is False
    assert caps.failure_reason is None
    assert caps.describe() == "No compatible GPU detected"


def test_probe_exception_isolation_all_channels_raise(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)  # import raises ImportError
    monkeypatch.setitem(sys.modules, "cupy", None)  # import raises ImportError

    def exploding_smi():
        raise RuntimeError("nvidia-smi exploded")

    monkeypatch.setattr(gpu_module, "_query_nvidia_smi", exploding_smi)

    caps = probe_gpu()  # must NOT propagate

    assert isinstance(caps, GpuCapabilities)
    assert caps.state == STATE_NO_GPU
    assert caps.backend_ready is False


def test_probe_backend_call_raises_is_isolated(monkeypatch):
    cupy = _fake_cupy_module()
    cupy.cuda.is_available = lambda: (_ for _ in ()).throw(
        RuntimeError("driver access failed")
    )
    cv2 = _fake_cv2_module(cuda_devices=1)
    cv2.cuda.getCudaEnabledDeviceCount = lambda: (_ for _ in ()).throw(
        RuntimeError("cv2 cuda exploded")
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    _no_nvidia_smi(monkeypatch)

    caps = probe_gpu()  # must NOT propagate

    assert isinstance(caps, GpuCapabilities)
    assert caps.backend_ready is False
    assert caps.gpu_detected is False
    assert caps.describe()  # non-empty


def test_probe_ready_cupy_path(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        _fake_cupy_module(
            device_name=b"Fake NVIDIA RTX", major=8, minor=9,
            vram_bytes=24 * 1024 ** 3,
        ),
    )
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_module())
    # nvidia-smi must not be consulted once a backend is ready.
    monkeypatch.setattr(gpu_module, "_query_nvidia_smi",
                        lambda: (_ for _ in ()).throw(AssertionError("unused")))

    caps = probe_gpu()

    assert caps.state == STATE_READY
    assert caps.gpu_detected is True
    assert caps.cuda_runtime_ready is True
    assert caps.cupy_ready is True
    assert caps.opencv_cuda_ready is False
    assert caps.backend_ready is True
    assert caps.device_name == "Fake NVIDIA RTX"
    assert caps.compute_capability == "8.9"
    assert caps.device_vram_mb == 24 * 1024
    assert caps.failure_reason is None
    description = caps.describe()
    assert "Fake NVIDIA RTX" in description
    assert "CUDA ready" in description


def test_probe_cupy_kernel_failure_is_backend_error(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        _fake_cupy_module(kernel_error=RuntimeError("CUDA headers missing")),
    )
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_module())
    _no_nvidia_smi(monkeypatch)

    caps = probe_gpu()

    assert caps.state == STATE_BACKEND_ERROR
    assert caps.gpu_detected is True
    assert caps.cuda_runtime_ready is True  # device visible to the runtime
    assert caps.cupy_ready is False
    assert caps.backend_ready is False
    assert caps.failure_reason is not None
    assert "cupy" in caps.failure_reason
    assert "CUDA headers missing" in caps.failure_reason
    # Device identity is still captured even when the kernel failed.
    assert caps.device_name == "Fake NVIDIA GPU"
    assert "backend unavailable" in caps.describe()


def test_probe_gpu_present_but_runtime_unavailable(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "cupy", _fake_cupy_module(available=False)
    )
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_module())
    monkeypatch.setattr(
        gpu_module, "_query_nvidia_smi", lambda: ("NVIDIA Fake MX150", 2048)
    )

    caps = probe_gpu()

    assert caps.state == STATE_GPU_NO_RUNTIME
    assert caps.gpu_detected is True
    assert caps.cuda_runtime_ready is False
    assert caps.cupy_ready is False
    assert caps.backend_ready is False
    assert caps.device_name == "NVIDIA Fake MX150"
    assert caps.device_vram_mb == 2048
    assert caps.failure_reason is not None
    assert "cupy" in caps.failure_reason
    assert caps.describe() == "GPU present but CUDA runtime unavailable"


def test_probe_gpu_present_but_no_python_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)  # CuPy absent
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_module())  # no cv2.cuda
    monkeypatch.setattr(
        gpu_module, "_query_nvidia_smi", lambda: ("NVIDIA Fake MX150", 2048)
    )

    caps = probe_gpu()

    assert caps.state == STATE_CUDA_NO_BACKEND
    assert caps.gpu_detected is True
    assert caps.cuda_runtime_ready is False
    assert caps.cupy_ready is False
    assert caps.opencv_cuda_ready is False
    assert caps.backend_ready is False
    assert caps.device_name == "NVIDIA Fake MX150"
    assert caps.device_vram_mb == 2048
    assert caps.failure_reason is not None
    assert "cupy" in caps.failure_reason
    assert "backend unavailable" in caps.describe()


def test_probe_ready_opencv_cuda_path(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_module(cuda_devices=2))
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(gpu_module, "_query_nvidia_smi",
                        lambda: (_ for _ in ()).throw(AssertionError("unused")))

    caps = probe_gpu()

    assert caps.state == STATE_READY
    assert caps.gpu_detected is True
    assert caps.cuda_runtime_ready is True
    assert caps.cupy_ready is False
    assert caps.opencv_cuda_ready is True
    assert caps.backend_ready is True
    assert caps.device_name is None
    assert "CUDA ready" in caps.describe()


# ---------------------------------------------------------------------------
# probe: unmocked coherence smoke (host-portable)
# ---------------------------------------------------------------------------


def test_probe_real_environment_never_raises_and_is_coherent():
    caps = probe_gpu()

    assert isinstance(caps, GpuCapabilities)
    assert caps.state in ALL_STATES
    assert caps.describe()  # non-empty
    # State machine coherence that must hold on every host.
    if caps.state == STATE_READY:
        assert caps.backend_ready is True
        assert caps.cuda_runtime_ready is True
        assert caps.cupy_ready or caps.opencv_cuda_ready
        assert caps.failure_reason is None
    elif caps.state == STATE_NO_GPU:
        assert caps.gpu_detected is False
        assert caps.backend_ready is False
    else:
        assert caps.gpu_detected is True
        assert caps.backend_ready is False


# ---------------------------------------------------------------------------
# acceleration policy
# ---------------------------------------------------------------------------


def test_policy_no_request_means_cpu():
    ready_caps = _caps(
        gpu_detected=True,
        cuda_runtime_ready=True,
        cupy_ready=True,
        backend_ready=True,
        state=STATE_READY,
        device_name="NVIDIA Fake MX150",
    )
    policy = AccelerationPolicy(ready_caps)  # request_gpu defaults to False
    policy_explicit = AccelerationPolicy(ready_caps, request_gpu=False)

    for resolved in (policy, policy_explicit):
        assert resolved.backend == "cpu"
        assert resolved.fallback_reason is None
        assert "CPU" in resolved.describe()


def test_policy_prefers_cupy_when_requested_and_ready():
    caps = _caps(
        gpu_detected=True,
        cuda_runtime_ready=True,
        cupy_ready=True,
        backend_ready=True,
        state=STATE_READY,
        device_name="NVIDIA Fake MX150",
    )
    policy = AccelerationPolicy(caps, request_gpu=True)

    assert policy.backend == "cupy"
    assert policy.fallback_reason is None
    assert "CuPy" in policy.describe()
    assert "NVIDIA Fake MX150" in policy.describe()


def test_policy_prefers_opencv_cuda_when_cupy_not_ready():
    caps = _caps(
        gpu_detected=True,
        cuda_runtime_ready=True,
        opencv_cuda_ready=True,
        backend_ready=True,
        state=STATE_READY,
        device_name="NVIDIA Fake MX150",
    )
    policy = AccelerationPolicy(caps, request_gpu=True)

    assert policy.backend == "opencv_cuda"
    assert policy.fallback_reason is None
    assert "OpenCV CUDA" in policy.describe()


def test_policy_falls_back_to_cpu_with_state_reason():
    caps = _caps()  # no_gpu
    policy = AccelerationPolicy(caps, request_gpu=True)

    assert policy.backend == "cpu"
    assert policy.fallback_reason == STATE_NO_GPU
    assert "CPU" in policy.describe()


def test_policy_fallback_reason_carries_backend_failure():
    caps = _caps(
        gpu_detected=True,
        cuda_runtime_ready=True,
        state=STATE_BACKEND_ERROR,
        failure_reason="cupy: real-kernel init failed (RuntimeError: CUDA headers missing)",
    )
    policy = AccelerationPolicy(caps, request_gpu=True)

    assert policy.backend == "cpu"
    assert policy.fallback_reason == caps.failure_reason
    description = policy.describe()
    assert "CPU" in description
    assert "CUDA headers missing" in description


def test_policy_describe_is_non_empty_for_all_branches():
    combos = [
        (_caps(state=STATE_READY, backend_ready=True, cupy_ready=True,
               gpu_detected=True), True),
        (_caps(state=STATE_READY, backend_ready=True, opencv_cuda_ready=True,
               gpu_detected=True), True),
        (_caps(state=STATE_NO_GPU), False),
        (_caps(state=STATE_NO_GPU), True),
        (_caps(state=STATE_BACKEND_ERROR, gpu_detected=True,
               failure_reason="cupy: boom"), True),
    ]
    for caps, request_gpu in combos:
        description = AccelerationPolicy(caps, request_gpu=request_gpu).describe()
        assert description and description.strip()


# ---------------------------------------------------------------------------
# capabilities describe()
# ---------------------------------------------------------------------------


def test_capabilities_describe_contains_device_name_when_known():
    caps = _caps(
        gpu_detected=True,
        cuda_runtime_ready=True,
        cupy_ready=True,
        backend_ready=True,
        state=STATE_READY,
        device_name="NVIDIA GeForce MX150",
    )
    description = caps.describe()
    assert "NVIDIA GeForce MX150" in description
    assert description == "NVIDIA GeForce MX150 — CUDA ready (CuPy)"


def test_capabilities_describe_fallback_without_device_name():
    caps = _caps(state=STATE_NO_GPU)
    assert caps.describe() == "No compatible GPU detected"
