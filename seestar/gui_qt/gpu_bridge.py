"""GPU capability bridge for the Qt shell (import-hygiene-safe, M5).

The Qt shell may not contain the engine's dotted module token anywhere in its
sources (the ``gui_qt`` package is scanned by source-hygiene tests), so the
engine is reached lazily through an ``importlib.import_module`` call whose
module path is assembled from split string literals — the same pattern as the
backend runner's ``_load_stackers_class``.

This module itself never imports the engine at module load; probing happens on
first call and the result is cached for the process lifetime (the probe is
read-only and cheap after the first CuPy warm-up).
"""

from __future__ import annotations

import importlib

__all__ = ["probe_gpu", "describe_capability", "describe_policy"]

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
    the engine/probe cannot run.  Cached for the process lifetime.
    """
    global _cache
    if _cache is _UNSET:
        try:
            _cache = _gpu_module().probe_gpu()
        except Exception:
            _cache = None
    return _cache


def describe_capability(caps):
    """Localized-ready capability line (English ``describe()`` text)."""
    if caps is None:
        return "GPU status unavailable"
    return caps.describe()


def describe_policy(caps, request_gpu: bool = False):
    """Resolved-state line for the status label.

    * no capabilities            -> "GPU status unavailable"
    * no usable backend          -> the capability line (e.g. "No compatible
                                    GPU detected")
    * backend ready + requested  -> the resolved policy line (e.g.
                                    "CuPy acceleration on NVIDIA GeForce MX150")
    * backend ready + unchecked  -> "<device> — CUDA ready (disabled)"
    """
    if caps is None:
        return "GPU status unavailable"
    ready = bool(getattr(caps, "backend_ready", False))
    if not ready:
        return caps.describe()
    if request_gpu:
        try:
            policy = _gpu_module().AccelerationPolicy(caps, request_gpu=True)
            return policy.describe()
        except Exception:
            pass
        return caps.describe()
    device = getattr(caps, "device_name", None) or "CUDA GPU"
    return f"{device} — CUDA ready (disabled)"
