"""Utilities to make Tkinter Variable classes resilient to background-thread GC.

Tkinter raises ``RuntimeError: main thread is not in main loop`` when a
``tk.Variable`` subclass is garbage-collected from a thread other than the one
that created the Tk application. On Windows this frequently happens at process
shutdown when background threads created by ``multiprocessing`` are still
running while Tkinter tears down.  The error is harmless but noisy for users.

This module monkey-patches the common ``Variable`` subclasses so their
``__del__`` methods silently ignore this specific ``RuntimeError`` while still
surfacing every other issue.
"""
from __future__ import annotations

import tkinter as tk
from typing import Type


def _make_thread_safe(cls: Type[tk.Variable]) -> Type[tk.Variable]:
    """Return a subclass of ``cls`` whose ``__del__`` ignores thread errors."""

    if getattr(cls, "_zemosaic_thread_safe", False):  # Already patched
        return cls

    class _ThreadSafeVariable(cls):  # type: ignore[misc]
        _zemosaic_thread_safe = True

        def __del__(self) -> None:  # pragma: no cover - exercised at shutdown
            try:
                super().__del__()
            except RuntimeError as exc:
                # Tkinter complains if a Variable is finalized off the Tk thread.
                if "main thread is not in main loop" not in str(exc):
                    raise

    _ThreadSafeVariable.__name__ = cls.__name__
    return _ThreadSafeVariable


def patch_tk_variables() -> None:
    """Monkey-patch Tkinter variable classes with thread-safe destructors."""

    for name in ("BooleanVar", "DoubleVar", "IntVar", "StringVar"):
        var_cls = getattr(tk, name, None)
        if var_cls is None:
            continue
        safe_cls = _make_thread_safe(var_cls)
        setattr(tk, name, safe_cls)


__all__ = ["patch_tk_variables"]
