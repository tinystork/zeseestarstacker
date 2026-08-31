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
* Spatial support is channel-invariant and exactly 2-D (H, W): one per-pixel
  support map per original exposure, independent of colour-channel count.

Design constraints honoured here
--------------------------------
1. Positive-only, finite, per-pixel support: negative / NaN / Inf /
   shape-mismatch / non-finite-squared inputs are rejected BEFORE any mutation.
2. Cumulative-overflow preflight: if EITHER candidate SUP_W1 or SUP_W2 would
   become non-finite after accumulation, the add is rejected BEFORE either
   array is mutated (atomic pair semantics; both stay byte/array identical).
3. Atomic pair mutation: a failed add() (or a failed restore) leaves both
   SUP_W1 and SUP_W2 byte/array unchanged.
4. No dependency on scientific WHT; no low-WHT gain; no science-pixel mutation;
   no rejection-mask semantics.
5. No batch/merge API: add() is the sole mutation, so decomposition invariance
   across partition markers (61 vs 3+17+41 vs 1+...+1) is exact by construction
   for an identical ordered sequence of per-exposure additions (same float64
   operation order).  There is deliberately no alternate merge API, so there is
   no alternate rounding order to document.
6. n_eff_support is a pure derived view that never mutates state; it uses an
   exact-first strategy (naive W1**2/W2 where the square stays finite) with an
   overflow-resistant fallback (W1/sqrt(W2))**2 for pixels whose square would
   overflow, so a finite W1 and W2 never produce a spurious undefined 0.0.

Dtype / memory rationale
------------------------
Accumulators default to numpy.float64.  N_eff_support uses the
overflow-resistant evaluation described above, so no finite SUP_W1/SUP_W2 pair
can silently degrade to an undefined result via the W1**2 term.  float64
provides ample precision headroom for the up-to-~100k-exposure target: a 100k
unit-weight witness is exact on this implementation.  float32 remains an
opt-in, memory-constrained option whose integer-exactness ceiling is 2**24
and which must be used only under a documented tolerance.

Bytes-per-pixel for the SUP_W1+SUP_W2 pair:

* float64: 16 bytes/pixel  (8 + 8)
* float32:  8 bytes/pixel  (4 + 4)

No hidden HWC duplication: the two accumulators are the only live arrays;
support is channel-invariant 2-D, so no per-channel copies are made here.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PositiveSupportAccumulator",
    "accumulate_support_pair",
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


def accumulate_support_pair(w1, w2, support, *, dtype=np.float64):
    """Atomically accumulate one positive support map into two external arrays.

    Mirrors PositiveSupportAccumulator.add() exactly, but mutates caller-provided
    (in-memory or memmap) float32/float64 (H, W) arrays in place.  This is the
    single source of truth for the atomic per-exposure support accumulation
    shared by the in-memory accumulator and the classic backend memmaps.

    Fail-before-mutation: a shape/dtype/finiteness/negativity/square-overflow or
    cumulative-overflow violation raises and leaves both arrays byte-identical.

    Parameters
    ----------
    w1, w2 : ndarray
        Existing (H, W) float accumulators (SUP_W1, SUP_W2) to mutate.
    support : array_like
        Per-pixel positive support map (H, W).
    dtype : numpy.dtype, optional
        Accumulator dtype (float32 or float64; default float64).
    """
    dtype = np.dtype(dtype)
    if dtype not in SUPPORT_DTYPES:
        raise TypeError(f"support dtype must be float32/float64, got {dtype.name!r}")
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    if w1.ndim != 2 or w2.ndim != 2:
        raise ValueError("support accumulators must be 2-D (H, W)")
    if w1.shape != w2.shape:
        raise ValueError("support accumulators must share a shape")
    if w1.dtype != dtype or w2.dtype != dtype:
        raise TypeError(
            f"support accumulators must have dtype {dtype}, got {w1.dtype}/{w2.dtype}"
        )
    if not (np.all(np.isfinite(w1)) and np.all(np.isfinite(w2))):
        raise ValueError("support accumulators contain non-finite samples")
    if np.any(w1 < 0.0) or np.any(w2 < 0.0):
        raise ValueError("support accumulators contain negative samples")
    shape = w1.shape
    s = np.asarray(support)
    if s.shape != shape:
        raise ValueError(f"support shape {s.shape} does not match accumulator shape {shape}")
    if not np.issubdtype(s.dtype, np.floating):
        s = s.astype(dtype)
    elif s.dtype != dtype:
        s = s.astype(dtype, copy=False)
    else:
        s = s.astype(dtype, copy=False)
    if not np.all(np.isfinite(s)):
        raise ValueError("support must be finite (NaN/Inf rejected)")
    if np.any(s < 0.0):
        raise ValueError("support must be non-negative")
    with np.errstate(over="ignore"):
        s2 = s * s
    if not np.all(np.isfinite(s2)):
        raise ValueError("support**2 overflowed to non-finite")
    with np.errstate(over="ignore", invalid="ignore"):
        new_w1 = w1 + s
        new_w2 = w2 + s2
    if not (np.all(np.isfinite(new_w1)) and np.all(np.isfinite(new_w2))):
        raise ValueError("cumulative support overflow: SUP_W1/SUP_W2 would become non-finite")
    w1[:] = new_w1
    w2[:] = new_w2


class PositiveSupportAccumulator:
    """Accumulate positive per-original-exposure support (SUP_W1 / SUP_W2).

    Parameters
    ----------
    shape : tuple of int
        Spatial (H, W) shape of the support maps.  Must be exactly 2-D with
        both dimensions positive.
    dtype : numpy.dtype, optional
        Accumulator dtype; must be float32 or float64 (default float64).

    The accumulator is channel-invariant: a single 2-D per-pixel support map
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
        if len(shape) != 2:
            raise ValueError(
                f"support shape must be exactly 2-D (H, W), got {shape!r}"
            )
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
    def add(self, support):
        """Accumulate one original exposure's positive per-pixel support.

        Atomic: both SUP_W1 and SUP_W2 are updated together only after the
        support map has fully validated AND the cumulative sums have been
        preflighted against overflow; a failed call leaves both unchanged.

        Parameters
        ----------
        support : array_like
            Per-pixel positive support map of shape (H, W) matching the
            accumulator.
        """
        accumulate_support_pair(self._w1, self._w2, support, dtype=self._dtype)

    # -------------------------------------------------------------- derived
    @property
    def n_eff_support(self):
        """Effective support N_eff_support = SUP_W1**2 / SUP_W2.

        Computed only where SUP_W2 > 0; 0.0 (documented neutral value) where
        undefined.  Non-negative and finite everywhere.  Never mutates state
        and returns an owned copy.

        Exact-first: naive W1**2 / W2 where the square stays finite (the common
        case, bit-exact), with an overflow-resistant fallback (W1 / sqrt(W2))**2
        for pixels whose W1**2 overflows.  Algebraically equal, so a finite W1
        and W2 never yield a spurious 0.0 from intermediate square overflow.
        """
        w1 = self._w1
        w2 = self._w2
        valid = w2 > 0.0
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            w1_sq = w1 * w1
            out = w1_sq / w2
            # Overflow-resistant fallback: (w1 / sqrt(w2))**2 == w1**2 / w2 but
            # never squares w1 directly.
            ratio = w1 / np.sqrt(w2)
            safe = ratio * ratio
        overflowed = ~np.isfinite(w1_sq)
        out = np.where(overflowed & valid, safe, out)
        # Clamp undefined / non-finite / negative to the documented neutral value.
        out = np.where(valid & np.isfinite(out) & (out >= 0.0), out, N_EFF_UNDEFINED_VALUE)
        return out

    # ------------------------------------------------------- snapshot/restore
    def to_state(self):
        """Export a canonical persistence-ready state snapshot (owned copies).

        The returned mapping carries version/shape/dtype metadata plus owned
        ndarray copies of SUP_W1/SUP_W2.  It is a canonical in-process
        restore source; it is NOT JSON-serializable as-is (arrays are kept as
        ndarrays to avoid bulky list serialization).
        """
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
