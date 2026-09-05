"""GPU capability bridge for the Qt shell (import-hygiene-safe, M5 + F3).

The Qt shell may not contain the engine's dotted module token anywhere in its
sources (the ``gui_qt`` package is scanned by source-hygiene tests), so the
engine is reached lazily through an ``importlib.import_module`` call whose
module path is assembled from split string literals — the same pattern as the
backend runner's ``_load_stackers_class``.

This module itself never imports the engine at module load; probing happens on
first call and the result is cached for the process lifetime (the probe is
read-only and cheap after the first CuPy warm-up).

The first probe (cold CuPy import + kernel JIT + nvidia-smi) can take
seconds, so it must never run on the Qt main thread: :class:`GpuProbeWorker`
is a minimal ``QThread`` that performs the probe in ``run()`` and delivers the
result through a queued ``resultReady`` signal.  Widget updates happen only in
the main-thread slot connected to that signal.
"""

from __future__ import annotations

import importlib

from PySide6.QtCore import QThread, Signal

__all__ = [
    "probe_gpu",
    "describe_capability",
    "describe_policy",
    "GpuProbeWorker",
]

_UNSET = object()
_cache = _UNSET  # GpuCapabilities | None | _UNSET (not probed yet)
_module = None  # lazily imported engine module (cached)


def _gpu_module():
    """Import the engine GPU module once (split-string, lazy)."""
    global _module
    if _module is None:
        _module = importlib.import_module(".".join(("seestar", "core", "gpu")))
    return _module


def probe_gpu():
    """Probe GPU capability through the engine probe; None on any failure.

    Returns a ``GpuCapabilities`` (with a ``.state`` attribute) or ``None`` if
    the engine/probe cannot run.  Cached for the process lifetime.  Thread-safe
    enough for the probe worker: the cache is filled at most once and reads of
    the cached object are atomic.
    """
    global _cache
    if _cache is _UNSET:
        try:
            _cache = _gpu_module().probe_gpu()
        except Exception:
            _cache = None
    return _cache


class GpuProbeWorker(QThread):
    """Run the (cold, potentially slow) GPU probe off the Qt main thread.

    ``run()`` executes the probe callable on the worker thread and emits the
    ``GpuCapabilities`` (or ``None``) through the queued ``resultReady``
    signal; the owner slot (living on the GUI thread) performs all widget
    updates.  The probe callable is injectable for deterministic tests; the
    default is the module-level :func:`probe_gpu` (which is process-cached, so
    only the first probe is slow).
    """

    resultReady = Signal(object)

    def __init__(self, probe_fn=None, parent=None):
        super().__init__(parent)
        self._probe_fn = probe_fn if probe_fn is not None else probe_gpu

    def run(self):  # noqa: D102 - Qt override name
        try:
            caps = self._probe_fn()
        except Exception:
            caps = None
        self.resultReady.emit(caps)


def describe_capability(caps):
    """Localized-ready capability line (English ``describe()`` text)."""
    if caps is None:
        return "GPU status unavailable"
    return caps.describe()


def describe_policy(caps, request_gpu: bool = False, boring: bool = False):
    """Resolved-state line for the status label.

    * no capabilities            -> "GPU status unavailable"
    * no usable backend          -> the capability line (e.g. "No compatible
                                    GPU detected")
    * boring mode + ready backend -> "GPU available — Boring default stack is
                                    CPU-only" (Boring always runs the default
                                    winsorized-sigma reduction, which is
                                    CPU-only; the status must not claim active
                                    GPU acceleration)
    * backend ready + requested  -> the resolved policy line (truthful
                                    wording, e.g. "GPU acceleration enabled —
                                    CuPy / NVIDIA GeForce MX150")
    * backend ready + unchecked  -> "<device> — GPU acceleration disabled"
    """
    if caps is None:
        return "GPU status unavailable"
    ready = bool(getattr(caps, "backend_ready", False))
    if not ready:
        return caps.describe()
    if boring:
        return "GPU available — Boring default stack is CPU-only"
    if request_gpu:
        try:
            policy = _gpu_module().AccelerationPolicy(caps, request_gpu=True)
            return policy.describe()
        except Exception:
            pass
        return caps.describe()
    device = getattr(caps, "device_name", None) or "CUDA GPU"
    return f"{device} — GPU acceleration disabled"
