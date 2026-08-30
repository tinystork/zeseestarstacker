"""Backend-neutral positive original-exposure support accumulator (COV-01A).

This module introduces an isolated, reusable representation of *positive*
original-exposure support that is independent from the scientific WHT
denominator.  It is the core state/math contract for the coverage-aware
peripheral-reconstruction effort and is deliberately NOT wired into
QueueManager, the classic/drizzle/reproject reducers, checkpoints, GUI,
render, or any existing scientific reducer in this subgate.

Contract
--------
For each original exposure i define a per-pixel positive support

    s_i = valid_geometric_support
          * optional_quality_significance
          * optional_spatial_support_taper

and accumulate

    SUP_W1 += s_i
    SUP_W2 += s_i**2

with the derived effective support

    N_eff_support = SUP_W1**2 / SUP_W2     where SUP_W2 > 0,
                    0.0                     otherwise (documented neutral value).

Semantics
---------
* This is a *support/confidence* domain, NOT the scientific estimator
  denominator and NOT a rejection-survivor count.  For rejection reducers the
  support confidence describes the *original geometric/quality* support and may
  deliberately differ from the surviving estimator WHT; no rejection mask is
  consumed here.
* SUP_W1 is a raw exposure count ONLY in the restricted unit-weight case
  (s_i in {0, 1} per pixel).  Nothing in this API reports SUP_W1 as a
  "count"; callers derive that interpretation themselves, if ever, only for
  the unit-weight case.

Design constraints honoured here
--------------------------------
1. Positive-only, finite, per-pixel support: negative / NaN / Inf /
   shape-mismatch / non-finite-squared inputs are rejected BEFORE any mutation.
2. Atomic pair mutation: a failed add() (or a failed restore) leaves both
   SUP_W1 and SUP_W2 byte/array unchanged.
3. No dependency on scientific WHT; no low-WHT gain; no science-pixel mutation;
   no rejection-mask semantics.
4. No batch/merge API: add() is the sole mutation, so decomposition invariance
   across partition markers (61 vs 3+17+41 vs 1+...+1) is exact by construction
   for an identical ordered sequence of per-exposure additions (same float64
   operation order).  There is deliberately no alternate merge API, so there is
   no alternate rounding order to document.
5. n_eff_support is a pure derived view: it never mutates state and returns
   0.0 wherever SUP_W2 is not positive.

Dtype / memory rationale
------------------------
Accumulators default to numpy.float64.  N_eff_support squares SUP_W1; float32
would lose integer exactness beyond 2**24 (~1.67e7) and would round SUP_W1**2
for large exposure counts, corrupting the ratio.  float64 keeps SUP_W1**2 exact
for unit-weight counts up to ~9.0e15 exposures and is the safe choice for the
up-to-~100k-exposure target.

Bytes-per-pixel for the SUP_W1+SUP_W2 pair:

* float64: 16 bytes/pixel  (8 + 8)
* float32:  8 bytes/pixel  (4 + 4) - opt-in only for memory-constrained,
  low-exposure-count, tolerance-documented use.

No hidden HWC duplication: the two accumulators are the only live arrays; a
colour image contributes the same per-pixel support map regardless of channel
count (support is channel-invariant by contract), so no per-channel copies are
made here.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PositiveSupportAccumulator",
    "SUPPORT_STATE_VERSION",
    "SUPPORT_DTYPES",
]

# Canonical snapshot/restore schema version for this domain.  Bump only on a
# breaking state-format change.
SUPPORT_STATE_VERSION = 1

# Supported accumulator dtypes (explicit, small allowlist).
SUPPORT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))

# Documented neutral value for N_eff_support where SUP_W2 is not positive.
N_EFF_UNDEFINED_VALUE = 0.0


class PositiveSupportAccumulator:
    """Accumulate positive per-original-exposure support (SUP_W1 / SUP_W2).

    Parameters
    ----------
    shape : tuple of int
        Spatial (H, W) shape of the support maps.  Must be >= 2-D with all
        dimensions positive.
    dtype : numpy.dtype, optional
        Accumulator dtype; must be float32 or float64 (default float64).

    The accumulator is channel-invariant: a single per-pixel support map
    represents the geometric/quality support of one original exposure,
    independent of colour-channel count.
    """

    def __init__(self, shape, *, dtype=np.float64):
        shape = self._normalize_shape(shape)
        dtype = np.dtype(dtype)
        if dtype not in SUPPORT_DTYPES:
            raise TypeError(
                f"support dtype must be one of {tuple(d.name for d in SUPPORT_DTYPES)}, "
                f"got {dtype.name!r}"
            )
        self._shape = shape
        self._dtype = dtype
        # The two live accumulation arrays (the only mutable state).
        self._w1 = np.zeros(shape, dtype=dtype)
        self._w2 = np.zeros(shape, dtype=dtype)

    # ------------------------------------------------------------------ shape
    @staticmethod
    def _normalize_shape(shape):
        if isinstance(shape, int):
            shape = (shape,)
        try:
            shape = tuple(int(v) for v in shape)
        except (TypeError, ValueError):
            raise ValueError(f"support shape must be a sequence of ints, got {shape!r}")
        if len(shape) < 2:
            raise ValueError(f"support shape must be >= 2-D, got {shape!r}")
        if any(v <= 0 for v in shape):
            raise ValueError(f"support shape dims must be positive, got {shape!r}")
        return shape

    # --------------------------------------------------------------- metadata
    @property
    def shape(self):
        """Spatial (H, W) shape of the support maps."""
        return self._shape

    @property
    def dtype(self):
        """Accumulator dtype."""
        return self._dtype

    # ------------------------------------------------------------ read access
    @property
    def support_w1(self):
        """SUP_W1 (sum of per-exposure support s_i), as an owned copy.

        This is a *support* accumulator, never exposed as an exposure count.
        """
        return self._w1.copy()

    @property
    def support_w2(self):
        """SUP_W2 (sum of per-exposure support squared s_i**2), as a copy."""
        return self._w2.copy()

    # -------------------------------------------------------------- mutation
    def _coerce_support(self, support):
        """Validate one per-pixel support map and return an owned float copy.

        Raises before any mutation on: non-array-like, shape mismatch,
        non-finite (NaN/Inf), or negative values.
        """
        s = np.asarray(support)
        if s.shape != self._shape:
            raise ValueError(
                f"support shape {s.shape} does not match accumulator shape "
                f"{self._shape}"
            )
        if not np.issubdtype(s.dtype, np.floating):
            s = s.astype(self._dtype)
        elif s.dtype != self._dtype:
            s = s.astype(self._dtype, copy=False)
        else:
            s = s.astype(self._dtype, copy=False)
        if not np.all(np.isfinite(s)):
            raise ValueError("support must be finite (NaN/Inf rejected)")
        if np.any(s < 0.0):
            raise ValueError("support must be non-negative")
        return s

    def add(self, support):
        """Accumulate one original exposure's positive per-pixel support.

        Atomic: both SUP_W1 and SUP_W2 are updated together only after the
        support map has fully validated; a failed call leaves both unchanged.

        Parameters
        ----------
        support : array_like
            Per-pixel positive support map of shape :attr:`shape`.
        """
        s = self._coerce_support(support)  # validates first (no mutation yet)
        with np.errstate(over="ignore"):
            s2 = s * s
        if not np.all(np.isfinite(s2)):
            # s is finite but s**2 overflowed; refuse before mutating.
            raise ValueError("support**2 overflowed to non-finite")
        # Commit atomically (no further validation can fail).
        self._w1 += s
        self._w2 += s2

    # -------------------------------------------------------------- derived
    @property
    def n_eff_support(self):
        """Effective support N_eff_support = SUP_W1**2 / SUP_W2.

        Computed only where SUP_W2 > 0; 0.0 (documented neutral value) where
        undefined.  Non-negative and finite everywhere.  Never mutates state
        and returns an owned copy.
        """
        w1 = self._w1
        w2 = self._w2
        valid = w2 > 0.0
        out = np.zeros(self._shape, dtype=self._dtype)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(w1 * w1, w2, out=out, where=valid)
        # Clamp any residual non-finite / negative ratio to the neutral value.
        out = np.where(np.isfinite(out) & (out >= 0.0), out, N_EFF_UNDEFINED_VALUE)
        return out

    # ------------------------------------------------------- snapshot/restore
    def to_state(self):
        """Export a canonical, JSON-friendly state snapshot (owned copies)."""
        return {
            "version": SUPPORT_STATE_VERSION,
            "shape": list(self._shape),
            "dtype": self._dtype.name,
            "support_w1": self._w1.copy(),
            "support_w2": self._w2.copy(),
        }

    @classmethod
    def from_state(cls, state):
        """Restore an accumulator from a canonical state snapshot.

        Validates the ENTIRE state (type, version, shape, dtype, and both
        arrays' shape/dtype/finiteness/non-negativity) before constructing any
        visible object.  The returned accumulator owns private copies of both
        arrays (never aliased to the caller's state).
        """
        if not isinstance(state, dict):
            raise TypeError(f"state must be a dict, got {type(state).__name__}")
        version = state.get("version")
        if version != SUPPORT_STATE_VERSION:
            raise ValueError(
                f"unsupported support state version {version!r} "
                f"(expected {SUPPORT_STATE_VERSION})"
            )
        shape = cls._normalize_shape(state.get("shape"))
        dtype = np.dtype(state.get("dtype"))
        if dtype not in SUPPORT_DTYPES:
            raise TypeError(
                f"state dtype must be one of {tuple(d.name for d in SUPPORT_DTYPES)}, "
                f"got {dtype.name!r}"
            )
        w1 = cls._validate_state_array(state.get("support_w1"), shape, dtype, "support_w1")
        w2 = cls._validate_state_array(state.get("support_w2"), shape, dtype, "support_w2")

        acc = cls.__new__(cls)
        acc._shape = shape
        acc._dtype = dtype
        acc._w1 = w1  # already an owned copy
        acc._w2 = w2  # already an owned copy
        return acc

    @staticmethod
    def _validate_state_array(arr, shape, dtype, name):
        """Validate one state array and return an owned copy (no aliasing)."""
        if arr is None:
            raise ValueError(f"state missing {name!r} array")
        a = np.asarray(arr)
        if a.shape != tuple(shape):
            raise ValueError(
                f"state {name!r} shape {a.shape} != declared shape {tuple(shape)}"
            )
        if a.dtype != dtype:
            raise TypeError(
                f"state {name!r} dtype {a.dtype} != declared dtype {dtype}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"state {name!r} contains non-finite samples")
        if np.any(a < 0.0):
            raise ValueError(f"state {name!r} contains negative samples")
        return np.array(a, dtype=dtype, copy=True)

    def __repr__(self):
        return (
            f"PositiveSupportAccumulator(shape={self._shape}, "
            f"dtype={self._dtype.name}, "
            f"n_positive={int(np.count_nonzero(self._w1 > 0))})"
        )

