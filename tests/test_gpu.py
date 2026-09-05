"""Tests for the GPU capability probe and acceleration policy.

Exercises :mod:`seestar.core.gpu` deterministically through mocks so the
suite is green on any host (with or without a GPU, with or without CuPy), plus
one unmocked coherence smoke test that only asserts the probe never raises and
returns a self-consistent :class:`GpuCapabilities`.  CuPy is the SOLE
production backend: OpenCV-CUDA detection is diagnostic-only (F1) and never
makes the capability ready or selectable.

Five states are each proven reachable:

* ``no_gpu``            -- no GPU evidence anywhere.
* ``gpu_no_runtime``    -- nvidia-smi sees a GPU, CuPy importable but its CUDA
                           runtime cannot reach the device.
* ``cuda_no_backend``   -- CUDA-capable hardware present but the CuPy
                           production backend unavailable (includes
                           OpenCV-CUDA-only machines, which are reported but
                           never ready).
* ``backend_error``     -- CuPy sees the device but the real kernel fails
                           (e.g. missing CUDA toolkit headers -> JIT failure).
* ``ready``             -- the CuPy production backend actually works.
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


def test_probe_opencv_cuda_only_is_diagnostic_not_ready(monkeypatch):
    """OpenCV-CUDA-only machines are NOT ready (F1: CuPy is the sole
    production backend); OpenCV CUDA is still reported diagnostically."""
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_module(cuda_devices=2))
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(gpu_module, "_query_nvidia_smi",
                        lambda: (_ for _ in ()).throw(AssertionError("unused")))

    caps = probe_gpu()

    assert caps.state == STATE_CUDA_NO_BACKEND
    assert caps.gpu_detected is True
    assert caps.cuda_runtime_ready is True  # a runtime sees the device
    assert caps.cupy_ready is False
    assert caps.opencv_cuda_ready is True  # still detected (diagnostic)
    assert caps.backend_ready is False  # OpenCV CUDA must NOT enable readiness
    assert caps.device_name is None
    assert caps.failure_reason is not None
    assert "CuPy" in caps.failure_reason or "cupy" in caps.failure_reason
    assert "backend unavailable" in caps.describe()
    assert "ready" not in caps.describe().lower()


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
        assert caps.cupy_ready is True  # ready means the CuPy backend works (F1)
        assert caps.failure_reason is None
    elif caps.state == STATE_NO_GPU:
        assert caps.gpu_detected is False
        assert caps.backend_ready is False
    else:
        assert caps.gpu_detected is True
        assert caps.backend_ready is False  # never ready without a CuPy kernel


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
    description = policy.describe()
    assert "GPU acceleration enabled" in description
    assert "CuPy" in description
    assert "NVIDIA Fake MX150" in description


def test_opencv_cuda_ready_never_makes_backend_ready():
    """F1: ``opencv_cuda_ready`` is diagnostic-only and does not enable the
    production backend, either at the capability or the policy level."""
    caps = _caps(
        gpu_detected=True,
        cuda_runtime_ready=True,
        cupy_ready=False,
        opencv_cuda_ready=True,
        backend_ready=False,
        state=STATE_CUDA_NO_BACKEND,
        device_name="NVIDIA Fake MX150",
    )
    assert caps.backend_ready is False
    policy = AccelerationPolicy(caps, request_gpu=True)
    assert policy.backend == "cpu"
    assert policy.fallback_reason is not None


def test_policy_backend_only_cpu_or_cupy():
    """F1: ``AccelerationPolicy.backend`` never resolves to ``opencv_cuda``
    across a matrix of capabilities."""
    matrix = [
        (_caps(), False),
        (_caps(), True),
        (_caps(state=STATE_READY, backend_ready=True, cupy_ready=True,
               gpu_detected=True), False),
        (_caps(state=STATE_READY, backend_ready=True, cupy_ready=True,
               gpu_detected=True), True),
        (_caps(state=STATE_CUDA_NO_BACKEND, gpu_detected=True,
               cupy_ready=False, opencv_cuda_ready=True, backend_ready=False,
               failure_reason="cupy absent; opencv diagnostic-only"), True),
        (_caps(state=STATE_GPU_NO_RUNTIME, gpu_detected=True,
               cupy_ready=False, backend_ready=False), True),
        (_caps(state=STATE_BACKEND_ERROR, gpu_detected=True,
               cupy_ready=False, backend_ready=False,
               failure_reason="cupy kernel failed"), True),
    ]
    for caps, request_gpu in matrix:
        backend = AccelerationPolicy(caps, request_gpu=request_gpu).backend
        assert backend in {"cpu", "cupy"}, (caps.state, request_gpu, backend)


def test_policy_opencv_cuda_alone_falls_back_to_cpu():
    """F1: a machine with only OpenCV-CUDA ready never selects a GPU backend."""
    caps = _caps(
        gpu_detected=True,
        cuda_runtime_ready=True,
        opencv_cuda_ready=True,
        backend_ready=False,
        state=STATE_CUDA_NO_BACKEND,
        failure_reason="cupy: absent; OpenCV-CUDA present but diagnostic-only",
    )
    policy = AccelerationPolicy(caps, request_gpu=True)

    assert policy.backend == "cpu"
    assert policy.fallback_reason == caps.failure_reason
    description = policy.describe()
    assert "CPU" in description
    assert "OpenCV CUDA acceleration" not in description


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
        (_caps(state=STATE_READY, backend_ready=True, cupy_ready=True,
               gpu_detected=True), False),
        (_caps(state=STATE_CUDA_NO_BACKEND, gpu_detected=True,
               opencv_cuda_ready=True, cupy_ready=False, backend_ready=False,
               failure_reason="cupy: absent"), True),
        (_caps(state=STATE_NO_GPU), False),
        (_caps(state=STATE_NO_GPU), True),
        (_caps(state=STATE_BACKEND_ERROR, gpu_detected=True,
               failure_reason="cupy: boom"), True),
    ]
    for caps, request_gpu in combos:
        description = AccelerationPolicy(caps, request_gpu=request_gpu).describe()
        assert description and description.strip()
        assert "OpenCV CUDA acceleration" not in description


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


def test_stacker_acceleration_policy_is_frozen():
    """F2: the stacker resolves its policy ONCE and freezes it — repeated
    access returns the SAME object, and mutating ``request_gpu`` after first
    resolution must not change the resolved backend."""
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    stacker = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    stacker._gpu_capabilities = _caps(
        gpu_detected=True,
        cuda_runtime_ready=True,
        cupy_ready=True,
        backend_ready=True,
        state=STATE_READY,
        device_name="NVIDIA Fake MX150",
    )
    stacker._acceleration_policy = None
    stacker.request_gpu = True

    first = stacker.acceleration_policy
    second = stacker.acceleration_policy
    assert first is second  # cached, not rebuilt per access
    assert first.backend == "cupy"

    # Later mutation of the intent field must NOT change the frozen backend.
    stacker.request_gpu = False
    assert stacker.acceleration_policy is first
    assert first.backend == "cupy"
    assert stacker.effective_backend == "cupy"


def test_stacker_policy_re_resolved_per_run():
    """R2-F1: each new ``start_processing`` re-resolves the policy from the
    CURRENT ``request_gpu`` (fresh policy per RUN), while the policy stays
    frozen for the duration of the active run.  Uses the real
    ``start_processing`` early-return path (no aligner -> returns False right
    after the freeze point), with ready fake capabilities and no real probe.
    """
    import logging

    from seestar.queuep.queue_manager import SeestarQueuedStacker

    ready_caps = _caps(
        gpu_detected=True,
        cuda_runtime_ready=True,
        cupy_ready=True,
        backend_ready=True,
        state=STATE_READY,
        device_name="NVIDIA Fake MX150",
    )

    def make_stacker(request_gpu):
        obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
        obj._gpu_capabilities = ready_caps
        obj._acceleration_policy = None
        obj._gpu_fallback_logged = set()
        obj.request_gpu = request_gpu
        obj.processing_active = False
        obj.logger = logging.getLogger("test_gpu")
        obj.update_progress = lambda *a, **k: None
        # ``aligner`` is intentionally absent: start_processing freezes the
        # policy, then bails out via the aligner guard (returns False) — no
        # heavy run machinery is reached.
        return obj

    # Run 1: GPU requested -> cupy.
    stacker = make_stacker(request_gpu=True)
    assert stacker.start_processing("/in", "/out") is False
    p1 = stacker.acceleration_policy
    assert p1.backend == "cupy"

    # Run 2: intent turned off -> a FRESH policy resolves to cpu.
    stacker.request_gpu = False
    assert stacker.start_processing("/in", "/out") is False
    p2 = stacker.acceleration_policy
    assert p2 is not p1
    assert p2.backend == "cpu"

    # Run 3: intent back on -> another fresh policy resolves to cupy.
    stacker.request_gpu = True
    assert stacker.start_processing("/in", "/out") is False
    p3 = stacker.acceleration_policy
    assert p3 is not p2
    assert p3.backend == "cupy"

    # Frozen for the duration of the run: mutating request_gpu after the last
    # start must not change the resolved backend.
    stacker.request_gpu = False
    assert stacker.acceleration_policy is p3
    assert p3.backend == "cupy"
    assert stacker.effective_backend == "cupy"


def test_refused_start_does_not_mutate_active_policy():
    """R3-F1: a refused concurrent Start must leave the ACTIVE run's policy
    object and backend IDENTICAL (the refusal check runs before the policy
    invalidate+re-resolve)."""
    import logging

    from seestar.queuep.queue_manager import SeestarQueuedStacker

    ready_caps = _caps(
        gpu_detected=True,
        cuda_runtime_ready=True,
        cupy_ready=True,
        backend_ready=True,
        state=STATE_READY,
        device_name="NVIDIA Fake MX150",
    )

    def make_stacker(request_gpu):
        obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
        obj._gpu_capabilities = ready_caps
        obj._acceleration_policy = None
        obj._gpu_fallback_logged = set()
        obj.request_gpu = request_gpu
        obj.processing_active = False
        obj.logger = logging.getLogger("test_gpu")
        obj.update_progress = lambda *a, **k: None
        return obj

    # Active CuPy run; a later Start with the intent OFF must be refused and
    # must NOT swap the active policy to CPU.
    stacker = make_stacker(request_gpu=True)
    p0 = stacker.acceleration_policy
    assert p0.backend == "cupy"

    stacker.processing_active = True
    stacker.request_gpu = False
    assert stacker.start_processing("/in", "/out") is False  # refused
    assert stacker.acceleration_policy is p0  # SAME object
    assert stacker.effective_backend == "cupy"  # unchanged

    # Inverse: active CPU policy; a refused Start with the intent ON must keep
    # it CPU.
    stacker2 = make_stacker(request_gpu=False)
    q0 = stacker2.acceleration_policy
    assert q0.backend == "cpu"

    stacker2.processing_active = True
    stacker2.request_gpu = True
    assert stacker2.start_processing("/in", "/out") is False  # refused
    assert stacker2.acceleration_policy is q0
    assert stacker2.effective_backend == "cpu"  # unchanged
