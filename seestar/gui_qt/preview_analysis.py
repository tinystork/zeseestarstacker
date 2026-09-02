"""Pure float preview-analysis core (ZSSS-OTPUX-PREVIEW-CORE-01) — display-only.

Toolkit-free float-domain analysis for the Qt preview pipeline.  This module
implements the ratified contracts in ``docs/output_truthfulness_preview_audit.md``:

* §5.2 — Option A: backend carries ``(legacy_normalized, raw_linear)``; Qt owns
  the stable/adaptive anchor mapping (p0.5 / p95, finite min/max fallback only
  when degenerate), so small changes preserve a fixed pixel's mapping while
  genuine photometric drift can widen the display range;
* §5.3 (PHI-R3 semantics, PHI-AUTO-HISTOGRAM-UX-V1 correction) — 512-bin
  float histogram over the **explicit plotting/bin range** ``[0, bin_hi]``
  (``bin_hi = max(1.0, robust plot high)`` — the full analysis upper when
  the top is dense or the sample small, otherwise the robust 99.5 %
  percentile of the sample, so a sparse extreme finite tail can never
  stretch the bins into a few widely spaced spikes; equals the display-level
  window ``[0, 1]`` when no headroom exists), ``log1p`` visualization
  counts, per-channel min/max/median/mean/std on the *exact same*
  deterministic sample, per-channel overflow counts (in-domain values above
  the bin high — the sparse tail is *never silently dropped*), robust plotted
  X range + explicit full analysis range metadata;

  **Model roles are explicit** (never conflated): ``range`` / ``full_range``
  declare the full preserved **analysis/control domain** ``(0, upper)``;
  ``bin_range`` declares the domain the 512 plotted bins actually live in;
  ``x_range`` is the robust **viewport** (auto zoom); ``overflow`` counts, per
  channel, the analysis values above the plotted bin high (their full extent
  stays visible in ``full_range`` and in the per-channel stats ``max``).
* §5.5 — Auto Stretch background-population algorithm (no min/max
  renormalization);
* §5.6 — Auto WB true-background-band algorithm.

It is deliberately *pure*:

* no Qt widgets / ``QImage`` (the legacy ``uint8`` ``QImage`` estimators remain
  in ``preview_adjust.py``; this module consumes/produces float ``ndarray``
  only);
* no scientific-engine imports (it never touches the science / alignment /
  enhancement / queue-manager modules);
* numpy is imported *lazily*, inside each function, so a fresh
  ``import seestar.gui_qt`` (or even ``import seestar.gui_qt.preview_analysis``)
  never pulls numpy into ``sys.modules``;
* every helper returns a fresh array and never mutates its input — the backend
  ``raw_linear`` array, the scientific accumulators and the caller's arrays are
  all left bit-identical.

Inputs are float arrays in one of two layouts: 2D mono ``(H, W)`` or 3D
channels-last ``(H, W, C)`` (RGB/RGBA uses the first three channels).  All
analysis algorithms operate on the **preserved analysis domain** produced by
the anchor mapping (PHI-R3):

* the anchor mapping and the WB derivation **preserve finite out-of-range
  float headroom** — they never hard-clip the bright tail to ``1.0``.  Only
  the final display-rendering boundary (the ``QImage``/``uint8`` conversion in
  :mod:`preview_render` / :mod:`preview_adjust`) bounds values to the display
  domain.  Sub-black mapped values (a pure anchor-mapping artifact below the
  display floor) are floored at ``0.0`` exactly as before;
* non-finite values never propagate: any NaN/Inf input (or an arithmetic
  overflow) maps to ``0.0`` at the anchor/WB boundaries, so analysis buffers
  are finite by construction;
* the float histogram/stats represent that preserved analysis range
  explicitly, with **distinct roles** (PHI-AUTO-HISTOGRAM-UX-V1): the full
  analysis range ``(0, upper)`` (``upper = max(1.0, global finite max)``)
  stays the stats/BP-WP-control truth; the 512 plotted bins live over the
  **bin/plot range** ``(0, bin_hi)`` (robust plot high, see
  :func:`_plot_bin_high`), so a sparse extreme finite tail is never binned
  into isolated spikes; in-domain values above the plotted top are counted
  per channel as ``overflow`` and stay visible through ``full_range`` and the
  per-channel stats ``max``.  FRP-H1 adds the **dual-domain** complement: the
  same 512-bin resolution also spans the **full domain** ``(0, upper)``
  (``full_counts``/``full_log_counts`` over ``full_hist_range``) from the
  *exact same* in-domain sample, so a Reset/Full view shows the complete
  sampled distribution — the sparse tail genuinely binned, never an empty
  widened axis.  The display-level window ``[0, 1]`` keeps full
  bin resolution whenever no headroom exists (bit-identical to the pre-R3
  ``[0, 1]`` model), and dense HDR headroom extends ``bin_hi`` so the bins
  and the per-channel stats describe the real preserved analysis values
  instead of silently dropping them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Ratified constants
# --------------------------------------------------------------------------

# §5.3 (PHI-R3 / PHI-AUTO-HISTOGRAM-UX-V1): exactly 512 bins over the
# plotting/bin range ``[0, bin_hi]`` (see :func:`compute_histogram_float`).
# ``bin_hi = max(1.0, min(upper, robust plot high))`` where the robust plot
# high is the 99.5 %-percentile of the concatenated in-domain analysis sample
# (the same deterministic percentile as the robust ``x_range`` viewport high),
# extended back to the full analysis upper when the top is *dense* (no sparse
# far tail — the max lies within ``SPARSE_TAIL_REL_GAP`` of the robust top) or
# the sample is too small for a 0.5 % tail to be meaningful (bit-identical
# legacy ``[0, 1]`` binning whenever no headroom exists).  ``HISTOGRAM_RANGE``
# remains the *display-level reference window* ``[0, 1]``.
HISTOGRAM_BINS = 512
HISTOGRAM_RANGE = (0.0, 1.0)

# §5.3 (PHI-AUTO-HISTOGRAM-UX-V1): relative value gap at which the analysis
# max is treated as a *sparse extreme tail* sitting far above the robust dense
# top (max > 1.25x the robust top).  When such a tail exists, the plotted bins
# end at the robust top and every value above it is counted as per-channel
# ``overflow`` metadata (never silently dropped — full range + stats ``max``
# stay the truthful analysis truth).
SPARSE_TAIL_REL_GAP = 0.25

# Analysis-domain floor (PHI-R3).  Mapped values below the display black level
# are a pure anchor-mapping artifact (they can never be displayed and carry no
# signal); :func:`map_raw_linear` floors them at exactly ``0.0`` and the
# histogram/stats sample excludes nothing above the floor, so counts and stats
# always describe the *same* finite non-negative analysis population.
ANALYSIS_DOMAIN_FLOOR = 0.0

# Histogram/stats upper-bound floor: the analysis range never ends below the
# display white level ``1.0``, so the display window keeps full bin resolution
# when no headroom exists.
HISTOGRAM_UPPER_FLOOR = 1.0

# §5.2: anchor separation epsilon.
ANCHOR_SEP = 1e-4

# §5.2 anchor percentiles.  The low anchor is the robust dark floor
# (p0.5); the high anchor is the *scene* top (p95), not the bright star
# tail (p99.5).  Anchoring the high end to the star tail makes the display
# mapping track the brightest pixels, which drift independently of the
# stable scene on deep stacks (S/N improvement, transient bright frames,
# exposure changes).  The monotonic drift ratchet then widens the span
# permanently, compressing the scene toward 0 in mapped space and driving
# Auto Stretch's black point to 0 on long runs (progressive darkening).
# Anchoring to the scene top keeps the mapping stable while the top
# (100 - ANCHOR_HI_PCT)% of pixels (stars) are allowed to saturate - normal
# and desirable for astro display.  Display-only.
ANCHOR_LO_PCT = 0.5
ANCHOR_HI_PCT = 95.0

# §5.2 (drift accommodation): hysteresis dead-band for the frozen-anchor
# display mapping.  A new raw frame re-anchors (widens) only when its robust
# percentile range (p0.5 / p95) has drifted beyond the frozen range by more
# than this fraction of the frozen span.  This keeps small frame-to-frame
# photometric evolution *stable* (anti-pumping) while a legitimate 2x-3x
# global drift widens the mapping before the preview white-outs.  Display-only;
# it never touches the scientific accumulators.
ANCHOR_DRIFT_HYSTERESIS = 0.25

# §5.2 / §5.3: deterministic sampling cap (documented).  At most this many
# pixels are fed to the percentile/median/histogram computations; larger arrays
# are subsampled with a *fixed stride* so results are deterministic.
MAX_SAMPLE_PIXELS = 1_000_000

# §5.5 Auto Stretch constants.
AUTO_STRETCH_MIN_SAMPLE = 20
AUTO_STRETCH_DEFAULTS = (0.01, 0.99)
AUTO_STRETCH_PCT_LO = 0.5
AUTO_STRETCH_PCT_BG = 60.0
AUTO_STRETCH_PCT_HI = 99.5
AUTO_STRETCH_BG_SPREAD_BP = 2.8
AUTO_STRETCH_BG_SPREAD_WP = 8.0
AUTO_STRETCH_MAD_SCALE = 1.4826

# §5.6 Auto WB constants.
AUTO_WB_SATURATION = 0.98
AUTO_WB_MIN_SAMPLE = 64
AUTO_WB_CENTRE_FLOOR = 1e-6
AUTO_WB_GAIN_MIN = 0.2
AUTO_WB_GAIN_MAX = 5.0
AUTO_WB_LUMA = (0.299, 0.587, 0.114)
NEUTRAL_WB = (1.0, 1.0, 1.0)

# §5.3 point 6: robust plotted X range percentiles.
X_RANGE_PCT_LO = 0.5
X_RANGE_PCT_HI = 99.5


def _load_numpy():
    """Lazily import numpy (the module object, or ``None`` when unavailable)."""
    try:
        import importlib

        return importlib.import_module("numpy")
    except Exception:
        return None


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _cap_sample(np: Any, vals, max_pixels: int = MAX_SAMPLE_PIXELS):
    """Deterministic fixed-stride subsample bounding ``vals`` to ``max_pixels``.

    ``vals`` is a 1D array in C (row-major) order.  When it is longer than the
    cap we keep every ``ceil(n / max_pixels)``-th element, so the result has at
    most ``max_pixels`` elements and is fully deterministic (no RNG).
    """
    n = int(vals.size)
    if n <= max_pixels:
        return vals
    stride = max(1, int(-(-n // max_pixels)))  # ceil(n / max_pixels)
    return vals[::stride]


def _finite_positive_sample(np: Any, arr):
    """Deterministic finite-positive sample (finite and > 0), capped/stride.

    Returns ``None`` when no finite-positive element exists.
    """
    vals = arr[np.isfinite(arr) & (arr > 0.0)]
    if vals.size == 0:
        return None
    return _cap_sample(np, vals)


def _luminance(np: Any, arr):
    """Rec.601 luminance of a mapped float buffer (2D or ``(H, W, 3+)``)."""
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return (
            AUTO_WB_LUMA[0] * arr[..., 0]
            + AUTO_WB_LUMA[1] * arr[..., 1]
            + AUTO_WB_LUMA[2] * arr[..., 2]
        )
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    return arr


def _as_float_array(np: Any, obj: Any) -> Optional[Any]:
    """Return ``obj`` as a validated 2D/3D float array, or ``None``."""
    try:
        arr = np.asarray(obj)
    except Exception:
        return None
    if arr.dtype.kind not in "fiu":
        return None
    if arr.ndim not in (2, 3):
        return None
    if arr.size == 0:
        return None
    return arr.astype(np.float64, copy=True)


def _analysis_channels(np: Any, arr):
    """Return ``[(name, 1D capped sample)]`` for the analysis buffer.

    RGB (>=3 channels) yields ``R``/``G``/``B`` (first three channels); mono 2D
    (or ``(H, W, 1)``) yields a single ``"L"`` channel.  The same deterministic
    capped sample is used for both the histogram and the per-channel stats
    (§5.3 point 4/5), so "same-sample" semantics hold by construction.
    """
    if arr.ndim == 2:
        return [("L", _cap_sample(np, arr.ravel()))]
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return [("L", _cap_sample(np, arr[..., 0].ravel()))]
        if arr.shape[2] >= 3:
            return [
                ("R", _cap_sample(np, arr[..., 0].ravel())),
                ("G", _cap_sample(np, arr[..., 1].ravel())),
                ("B", _cap_sample(np, arr[..., 2].ravel())),
            ]
    return []


def _sample_stats(np: Any, sample):
    """Per-channel ``{min, max, median, mean, std}`` over the given sample."""
    return {
        "min": float(np.min(sample)),
        "max": float(np.max(sample)),
        "median": float(np.median(sample)),
        "mean": float(np.mean(sample)),
        "std": float(np.std(sample)),
    }


def _analysis_sample(np: Any, sample):
    """Finite non-negative values of a capped sample (PHI-R3 analysis domain).

    The histogram counts and the per-channel stats must describe the *exact
    same* finite analysis sample (§5.3 point 5): non-finite values and values
    below the analysis floor (``0.0`` — the sub-black display floor) are
    excluded here, so ``np.histogram`` never silently drops a value that the
    stats would otherwise describe.  Values **above** ``1.0`` (preserved HDR
    headroom) are kept: they are part of the analysis population and are
    represented by the bins above the display window.
    """
    return sample[np.isfinite(sample) & (sample >= ANALYSIS_DOMAIN_FLOOR)]


def _analysis_upper(np: Any, channels) -> float:
    """Deterministic analysis upper bound ``upper = max(1.0, finite max)``.

    The global finite maximum over every channel's capped sample, floored at
    the display white level ``1.0`` so the display window always has full bin
    resolution when no headroom exists.  Returns ``HISTOGRAM_UPPER_FLOOR``
    when no usable sample exists (the caller fail-closes anyway).
    """
    upper = HISTOGRAM_UPPER_FLOOR
    for _, sample in channels:
        in_domain = _analysis_sample(np, sample)
        if in_domain.size:
            upper = max(upper, float(np.max(in_domain)))
    return upper


def analysis_upper_bound(mapped) -> float:
    """Analysis domain upper bound of a buffer: ``max(1.0, finite max)``.

    Public seam (PHI-R3.1) so GUI-thread callers (display BP/WP control
    domain, marker domain, auto-stretch reconcile) use the *exact same*
    deterministic value as the histogram model's ``range``/``full_range``
    upper computed by :func:`compute_histogram_float` (same per-channel capped
    samples), keeping the analysis axis, the controls and the model in
    agreement.  Returns ``HISTOGRAM_UPPER_FLOOR`` (``1.0``) for missing /
    unusable input.
    """
    np = _load_numpy()
    if np is None:
        return HISTOGRAM_UPPER_FLOOR
    arr = np.asarray(mapped, dtype=np.float64)
    if arr.size == 0 or arr.ndim not in (2, 3):
        return HISTOGRAM_UPPER_FLOOR
    channels = _analysis_channels(np, arr)
    if not channels:
        return HISTOGRAM_UPPER_FLOOR
    return _analysis_upper(np, channels)


def _robust_x_range_from_samples(np: Any, channels, upper: float = HISTOGRAM_UPPER_FLOOR) -> Tuple[float, float]:
    """Robust plotted X range from the *finite* part of the channel samples.

    Guarded against empty / degenerate samples: returns the full analysis
    range ``(0, upper)`` when nothing usable remains.  The percentiles are
    computed on the analysis sample (finite, ``>= 0``), so with dense HDR
    headroom the robust high end can legitimately exceed ``1.0``.
    """
    parts = []
    for _, sample in channels:
        in_domain = _analysis_sample(np, sample)
        if in_domain.size:
            parts.append(in_domain)
    if not parts:
        return (ANALYSIS_DOMAIN_FLOOR, upper)
    all_vals = np.concatenate(parts)
    lo = float(np.percentile(all_vals, X_RANGE_PCT_LO))
    hi = float(np.percentile(all_vals, X_RANGE_PCT_HI))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return (ANALYSIS_DOMAIN_FLOOR, upper)
    return (lo, hi)


def _robust_top_value(np: Any, vals, pct: float = X_RANGE_PCT_HI) -> float:
    """Robust top of a 1D value array: ``max(1.0, percentile(vals, pct))``.

    The percentile is computed with numpy's default linear interpolation over
    the (finite, non-negative) values; the ``1.0`` floor keeps the
    display-window top as the minimum robust top (legacy ``[0, 1]`` buffers
    therefore keep a robust top of exactly ``1.0``, and the caller's clip at
    that value stays bit-identical).  Deterministic (no RNG); never mutates
    ``vals``.  Returns ``1.0`` for an empty input.
    """
    if vals is None or int(vals.size) == 0:
        return 1.0
    return max(1.0, float(np.percentile(vals, pct)))


def _plot_bin_high(np: Any, channels, upper: float) -> float:
    """Deterministic plotting/bin-range high of an analysis buffer.

    PHI-AUTO-HISTOGRAM-UX-V1 semantics — the returned value is the X-axis
    upper bound the 512 bins are spread over:

    * **No headroom** (``upper == 1.0``) or a **dense top** (the finite max
      is within ``SPARSE_TAIL_REL_GAP`` above the robust 99.5 % top, e.g. a
      continuous population that really reaches its max): the full analysis
      upper is kept, so the model is bit-identical to the legacy/R3
      full-range binning (zero overflow) — a genuine broad dynamic range
      keeps every bin dense because the population itself spans it;
    * **Sparse extreme far tail** (a few values far above the dense body,
      e.g. hot pixels at 282 while the bulk ends at ~2.4): the plot high is
      pulled down to the robust top percentile (the same 99.5 % percentile as
      the auto-zoom viewport high), so the auto/default visible window holds
      most of the 512 bins instead of ~4 widely spaced spikes.  Values above
      the plot high are counted per channel as ``overflow`` metadata and stay
      visible through ``full_range`` and the stats ``max``.

    Small samples (``<= 512`` in-domain values across the channels — fewer
    than one value per bin) never trigger the sparse-tail cut: with so few
    samples a single top value is not a statistically distinguishable "sparse
    tail", so the full analysis range is binned exactly as before
    (deterministic, documented).  Returns ``upper`` when no usable sample
    exists.
    """
    all_vals = _concat_in_domain(np, channels)
    n = int(all_vals.size)
    if n == 0:
        return float(upper)
    # A 0.5 % top tail is only meaningful when it represents >= 1 sample.
    if n <= 512:
        return float(upper)
    robust_top = _robust_top_value(np, all_vals, X_RANGE_PCT_HI)
    finite_max = float(np.max(all_vals))
    if robust_top >= float(upper) - 1e-12:
        # Dense/continuous top reaching the analysis upper (or no headroom):
        # keep the full-range binning (bit-identical legacy behaviour).
        return float(upper)
    if finite_max <= robust_top * (1.0 + SPARSE_TAIL_REL_GAP):
        # The max is at most 25% above the robust top: dense enough to bin.
        return float(upper)
    # Sparse extreme far tail: plot up to the robust top percentile; values
    # above are overflow metadata (truthful via full_range/stats max).
    return min(float(upper), robust_top)


def _histogram_overflow(np: Any, in_domain, bin_upper: float) -> int:
    """In-domain values strictly above the plotting/bin high (overflow).

    ``np.histogram`` silently drops values outside its ``range``; this count
    makes the drop explicit and truthful (the values remain visible through
    the stats ``max`` and the full analysis range metadata).  Returns ``0``
    whenever the bin high equals the analysis upper (nothing is above it).
    """
    if bin_upper >= float(np.max(in_domain)):
        return 0
    return int(np.count_nonzero(in_domain > bin_upper))


def _concat_in_domain(np: Any, channels) -> Any:
    """Concatenate the per-channel analysis samples (finite, >= floor).

    The channels' in-domain samples (same deterministic capped samples used
    for the per-channel counts/stats) are concatenated into one array so the
    shared X axis (bin range + robust viewport) is derived from the exact
    same population the counts describe.  Returns an empty array when no
    channel contributes a usable sample.
    """
    parts = []
    for _, sample in channels:
        in_domain = _analysis_sample(np, sample)
        if in_domain.size:
            parts.append(in_domain)
    if not parts:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(parts)


# --------------------------------------------------------------------------
# §5.2 — Option A payload extraction + stable/adaptive anchor mapping
# --------------------------------------------------------------------------

def extract_raw_linear(data: Any) -> Optional[Any]:
    """Extract an immutable raw-linear float array from a preview payload.

    Option A payloads are tuples ``(legacy_normalized, raw_linear)``: the
    *second* element is returned as an independent float copy.  Legacy
    single-array payloads (a lone array, or a tuple whose second element is not
    a 2D/3D array) fall back to a copy of the first/only array, so old
    producers keep working.  Returns ``None`` for missing / non-array data.

    Empty tuples/lists and one-element malformed sequences return ``None``
    without raising.
    """
    np = _load_numpy()
    if np is None:
        return None
    arr = None
    if isinstance(data, (tuple, list)):
        if len(data) >= 2:
            arr = _as_float_array(np, data[1])
        if arr is None and len(data) >= 1:
            arr = _as_float_array(np, data[0])
    else:
        arr = _as_float_array(np, data)
    return arr  # already a fresh float64 copy


def compute_anchors(raw_linear, sep: float = ANCHOR_SEP) -> Tuple[float, float]:
    """Compute fixed normalization anchors ``(lo, hi)`` from a raw-linear array.

    §5.2 Option A: anchors come from a deterministic finite-positive sample
    (``percentile(sample, ANCHOR_LO_PCT)`` / ``percentile(sample, ANCHOR_HI_PCT)``), falling back
    to the finite min/max *only* when that sample is degenerate (empty,
    non-finite, or ``hi <= lo + sep``).  Always returns ``(lo, hi)`` with
    ``hi > lo`` so the mapping is non-degenerate.
    """
    np = _load_numpy()
    if np is None:
        return (0.0, 1.0)
    arr = np.asarray(raw_linear, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 1.0)

    sample = _finite_positive_sample(np, arr)
    if sample is not None and sample.size > 0:
        lo = float(np.percentile(sample, ANCHOR_LO_PCT))
        hi = float(np.percentile(sample, ANCHOR_HI_PCT))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo + sep:
            return (lo, hi)

    # Degenerate percentile sample: fall back to the finite min/max.
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi - lo > sep:
        return (lo, hi)
    # Constant finite image: widen symmetrically so the value maps to 0.5.
    mid = 0.5 * (lo + hi)
    return (mid - sep, mid + sep)


def map_raw_linear(raw_linear, anchor_lo: float, anchor_hi: float) -> Any:
    """Map raw-linear values through frozen anchors into the analysis domain.

    ``mapped = (raw - lo) / (hi - lo)`` using the same anchors across
    successive previews (§5.2 regression: a fixed raw pixel maps identically
    when later-frame extrema change).  PHI-R3: the mapping **preserves finite
    out-of-range float headroom** — values above the display white level
    ``1.0`` (raw signal beyond the high anchor) are *not* hard-clipped here;
    only the final display-rendering boundary (the ``QImage``/``uint8``
    conversion) clamps to the display domain.

    Domain guarantees:

    * finite mapped values ``> 0`` are kept verbatim (including ``> 1``
      headroom);
    * mapped values ``<= 0`` (the sub-black tail below the low anchor, which
      can never be displayed and carries no signal) floor to exactly ``0.0``,
      bit-identical to the pre-R3 low-side clip;
    * non-finite results (NaN/Inf input, or a finite input whose mapping
      overflows float64) become ``0.0`` — **no NaN/Inf ever propagates** into
      the analysis buffers.

    Non-mutating (always returns a fresh array).  For drift accommodation,
    callers re-anchor via :func:`adapt_anchors_for_drift` before calling this
    mapping.
    """
    np = _load_numpy()
    if np is None:
        return None
    arr = np.asarray(raw_linear, dtype=np.float64)
    lo = float(anchor_lo)
    hi = float(anchor_hi)
    denom = hi - lo
    if denom <= 0.0:
        denom = ANCHOR_SEP
    with np.errstate(invalid="ignore", divide="ignore"):
        mapped = (arr - lo) / denom
    # Floor sub-black values and sanitize non-finite results; finite positive
    # values (including headroom > 1) are preserved bit-exactly.
    return np.where(np.isfinite(mapped) & (mapped > 0.0), mapped, 0.0)


def adapt_anchors_for_drift(
    anchor_lo: float,
    anchor_hi: float,
    raw_linear,
    hysteresis: float = ANCHOR_DRIFT_HYSTERESIS,
    sep: float = ANCHOR_SEP,
) -> Tuple[float, float]:
    """Return the effective frozen anchors for a new raw-linear frame.

    §5.2 drift accommodation (display-only).  The frozen anchors are kept
    bit-identical while the new frame's robust percentile range (p0.5 / p95)
    stays within a hysteresis band around them — so a fixed raw pixel keeps
    mapping identically across small (frame-to-frame) evolution, preserving the
    anti-pumping intent.  When the new frame's robust range has drifted beyond
    that band, the anchors are widened **monotonically outward** (a ratchet:
    ``lo`` only decreases, ``hi`` only increases) to cover the new robust
    range, so a legitimate 2x-3x photometric drift no longer maps the bulk of
    the image to exactly ``1.0`` (artificial saturation / white-out).

    Monotonicity means the mapping can only "zoom out" across a context, never
    oscillate: once widened for a brighter stack, a transient dimmer frame does
    not shrink the range back (strong temporal anti-pumping).  The expansion is
    *bounded* — it is driven by the new frame's robust p0.5/p99.5 (never by a
    single outlier pixel) and never overshoots the data.  Returns a fresh
    ``(lo, hi)`` pair with ``hi > lo``; the input array is never mutated.

    A degenerate new frame (no finite-positive sample) carries no drift
    information and leaves the anchors unchanged.
    """
    np = _load_numpy()
    if np is None:
        return (float(anchor_lo), float(anchor_hi))
    arr = np.asarray(raw_linear, dtype=np.float64)
    if arr.size == 0:
        return (float(anchor_lo), float(anchor_hi))

    lo = float(anchor_lo)
    hi = float(anchor_hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        # No usable frozen anchors: fall back to a fresh anchor computation.
        return compute_anchors(arr, sep=sep)

    sample = _finite_positive_sample(np, arr)
    if sample is None or sample.size == 0:
        return (lo, hi)
    cur_lo = float(np.percentile(sample, ANCHOR_LO_PCT))
    cur_hi = float(np.percentile(sample, ANCHOR_HI_PCT))
    if not (np.isfinite(cur_lo) and np.isfinite(cur_hi)):
        return (lo, hi)

    span = hi - lo
    if span <= 0.0:
        span = sep
    band = float(hysteresis) * span

    new_lo = lo
    new_hi = hi
    # Bright drift: the robust high tail escapes the frozen range -> widen up.
    if cur_hi > hi + band:
        new_hi = cur_hi
    # Dark drift: the robust low tail escapes the frozen range -> widen down.
    if cur_lo < lo - band:
        new_lo = cur_lo

    # Never return a degenerate pair; symmetric widening is a last-resort guard
    # and only triggers when the frozen anchors were already degenerate.
    if new_hi - new_lo <= sep:
        mid = 0.5 * (new_lo + new_hi)
        new_lo = mid - sep
        new_hi = mid + sep
    return (new_lo, new_hi)


# --------------------------------------------------------------------------
# §5.3 — WB-only derivation + float histogram / stats / X range
# --------------------------------------------------------------------------

def apply_wb_float(mapped, wb=NEUTRAL_WB) -> Any:
    """Apply white-balance gains to a mapped analysis buffer.

    Produces the WB-only analysis buffer (§5.3): per-channel multiply by the
    R/G/B gains.  PHI-R3: the gains are **not clipped to ``[0, 1]``** — a
    strong gain applied to in-range signal (or to preserved anchor headroom)
    legitimately produces analysis values ``> 1``, which the histogram/stats
    must see.  Only the final display-rendering boundary bounds values to the
    display domain.

    Domain guarantees (mirror :func:`map_raw_linear`): finite results are kept
    verbatim; results ``<= 0`` floor to ``0.0``; non-finite results (NaN/Inf
    input or overflow) become ``0.0``, so the WB-only buffer stays finite.
    Mono (2D) data is unaffected and returned as a plain copy.  Always returns
    a fresh array; the input is never mutated.
    """
    np = _load_numpy()
    if np is None:
        return None
    arr = np.asarray(mapped, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr.copy()
    r, g, b = (float(wb[0]), float(wb[1]), float(wb[2]))
    out = arr.copy()
    for i, gain in enumerate((r, g, b)):
        scaled = arr[..., i] * gain
        out[..., i] = np.where(
            np.isfinite(scaled) & (scaled > 0.0), scaled, 0.0
        )
    return out


def compute_histogram_float(mapped) -> Optional[Dict[str, Any]]:
    """Compute the §5.3 float histogram + stats from an analysis buffer.

    PHI-R3 semantics: the model represents the **preserved analysis float
    range**, not a pre-clipped ``[0, 1]`` domain.  PHI-AUTO-HISTOGRAM-UX-V1
    makes the *plot/bin* domain an explicit, distinct role: the 512 bins are
    spread over a **robust bin range** ``[0, bin_hi]`` so the auto/default
    visible window keeps dense bins even when a sparse extreme finite tail
    exists, while the **full analysis range** ``(0, upper)`` and the per-
    channel stats (including the tail ``max``) remain the truthful full-
    domain metadata.  Returns a dict with:

    * ``bins`` — ``HISTOGRAM_BINS`` (512) bins over the **bin range**
      (fixed by contract; there is no per-call bin override);
    * ``range`` / ``full_range`` — the explicit full analysis/control domain
      ``(0.0, upper)`` where ``upper = max(1.0, finite max)`` over the whole
      buffer: equal to the display-level window ``(0.0, 1.0)`` when no
      headroom exists, extended past ``1.0`` when the analysis/WB buffer
      carries finite HDR headroom — the stats and BP/WP control domain truth
      (a sparse extreme tail never shrinks it);
    * ``bin_range`` — the plotting/bin domain ``(0.0, bin_hi)`` the 512
      ``counts``/``log_counts`` bins actually live in: ``bin_hi`` is the
      deterministic robust plot high (see :func:`_plot_bin_high` — the full
      analysis upper when the top is dense or the sample small, otherwise the
      robust 99.5 % top percentile), so a sparse extreme tail is *not*
      binned into a few isolated spikes;
    * ``overflow`` — per-channel count of in-domain values strictly above
      ``bin_hi`` (their presence/full extent stays truthful via ``stats``
      ``max`` and ``full_range``; nothing is silently dropped), plus
      ``overflow_total``;
    * ``channels`` — ``["L"]`` (mono) or ``["R", "G", "B"]``;
    * ``counts`` — per-channel ``int64`` bin counts over ``bin_range``;
    * ``log_counts`` — ``log1p(counts)`` visualization counts (empty bin == 0);
    * ``full_counts`` / ``full_log_counts`` — per-channel ``int64`` bin counts
      (resp. ``log1p`` visualization counts) of the **full-domain histogram**
      (FRP-H1): 512 bins over the full analysis range ``(0.0, upper)``, i.e.
      the *complete* sampled distribution including any sparse extreme tail
      (the tail that ``counts`` leaves as overflow is genuinely binned here, so
      a Reset/Full view shows the real tail population, not an empty widened
      axis).  They are computed from the **exact same** deterministic
      in-domain sample as ``counts``/``overflow`` (same arrays, no second
      image traversal), so per channel ``sum(full_counts) == sampled_count``
      while ``sum(counts) + overflow == sampled_count``.  Degenerate-identical
      to ``counts``/``log_counts`` when ``bin_hi == upper`` (no sparse tail:
      both histograms bin the same domain — documented, harmless);
    * ``full_hist_range`` — the X domain the full-domain bars live in:
      ``(0.0, upper)``, identical to ``range``/``full_range``;
    * ``stats`` — per-channel ``{min, max, median, mean, std}`` computed on the
      *exact same* deterministic in-domain sample as ``counts`` + overflow
      (so with headroom the per-channel ``max`` truthfully reports the
      preserved analysis peak, including values above the plotted top);
    * ``x_range`` — robust plotted X range (percentile-based on the analysis
      sample; always at or below the bin high, so auto zoom stays inside the
      binned domain; can exceed ``1.0`` when headroom is dense).

    The histogram counts, the overflow counts, the full-domain counts and all
    five stats are computed over the *exact same* finite non-negative analysis
    sample (non-finite values and sub-black values are excluded from all of
    them).  When a required channel has no usable sample the analysis fails
    closed and returns ``None`` — it never fabricates synthetic pixels.

    Returns ``None`` for missing / unusable input.
    """
    np = _load_numpy()
    if np is None:
        return None
    arr = np.asarray(mapped, dtype=np.float64)
    if arr.size == 0 or arr.ndim not in (2, 3):
        return None
    channels = _analysis_channels(np, arr)
    if not channels:
        return None
    # Fail-closed pre-check: every required channel must contribute a usable
    # analysis sample (finite, non-negative); the in-domain samples are kept
    # so counts, overflow and stats describe the exact same population.
    in_domain_by_channel: Dict[str, Any] = {}
    for name, sample in channels:
        in_domain = _analysis_sample(np, sample)
        if in_domain.size == 0:
            # Never fabricate a synthetic pixel for an unusable required
            # channel (all-NaN / sub-black channel).
            return None
        in_domain_by_channel[name] = in_domain
    upper = _analysis_upper(np, channels)
    analysis_range = (ANALYSIS_DOMAIN_FLOOR, upper)
    bin_hi = _plot_bin_high(
        np, [(name, in_domain_by_channel[name]) for name, _ in channels], upper
    )
    bin_range = (ANALYSIS_DOMAIN_FLOOR, bin_hi)
    counts: Dict[str, Any] = {}
    log_counts: Dict[str, Any] = {}
    full_counts: Dict[str, Any] = {}
    full_log_counts: Dict[str, Any] = {}
    stats: Dict[str, Dict[str, float]] = {}
    overflow: Dict[str, int] = {}
    overflow_total = 0
    for name, _sample in channels:
        in_domain = in_domain_by_channel[name]
        hist, _ = np.histogram(
            in_domain, bins=HISTOGRAM_BINS, range=bin_range
        )
        hist = hist.astype(np.int64)
        counts[name] = hist
        log_counts[name] = np.log1p(hist.astype(np.float64))
        # FRP-H1 full-domain histogram: 512 bins over the true analysis
        # maximum (0.0, upper), from the *exact same* in-domain array as the
        # robust ``counts`` above (no second image traversal / sample).  This
        # keeps the complete sampled distribution binned — the sparse extreme
        # tail that ``counts`` leaves as overflow is a real population here —
        # so a Reset/Full view displays genuine tail bars over the full domain
        # instead of an empty widened axis.  When ``bin_hi == upper`` (dense
        # top / small sample / no headroom) the two histograms are
        # degenerate-identical (same bins over the same range) — fine.
        full_hist, _ = np.histogram(
            in_domain, bins=HISTOGRAM_BINS, range=analysis_range
        )
        full_hist = full_hist.astype(np.int64)
        full_counts[name] = full_hist
        full_log_counts[name] = np.log1p(full_hist.astype(np.float64))
        stats[name] = _sample_stats(np, in_domain)
        n_overflow = _histogram_overflow(np, in_domain, bin_hi)
        overflow[name] = n_overflow
        overflow_total += n_overflow
    return {
        "bins": HISTOGRAM_BINS,
        "range": analysis_range,
        "channels": [name for name, _ in channels],
        "counts": counts,
        "log_counts": log_counts,
        "full_counts": full_counts,
        "full_log_counts": full_log_counts,
        "full_hist_range": analysis_range,
        "stats": stats,
        "x_range": _robust_x_range_from_samples(np, channels, upper),
        "full_range": analysis_range,
        "bin_range": bin_range,
        "overflow": overflow,
        "overflow_total": overflow_total,
    }


def compute_histogram_stats_float(mapped) -> Optional[Dict[str, Dict[str, float]]]:
    """Per-channel ``{min, max, median, mean, std}`` from the §5.3 sample.

    Thin wrapper over :func:`compute_histogram_float` so the stats are always
    derived from the *exact same* deterministic sample as the histogram counts.
    """
    result = compute_histogram_float(mapped)
    if result is None:
        return None
    return result["stats"]


def compute_robust_x_range(mapped) -> Tuple[float, float]:
    """Robust plotted X range for the analysis buffer (§5.3 point 6).

    Percentile-based over the finite non-negative analysis sample, guarded
    against empty/degenerate samples; the explicit full analysis range is the
    caller's toggle.  With dense HDR headroom the robust high end can
    legitimately exceed ``1.0``.
    """
    np = _load_numpy()
    if np is None:
        return (0.0, 1.0)
    arr = np.asarray(mapped, dtype=np.float64)
    if arr.size == 0 or arr.ndim not in (2, 3):
        return (0.0, 1.0)
    channels = _analysis_channels(np, arr)
    if not channels:
        return (0.0, 1.0)
    return _robust_x_range_from_samples(np, channels, _analysis_upper(np, channels))


# --------------------------------------------------------------------------
# §5.5 — Auto Stretch (background-population algorithm)
# --------------------------------------------------------------------------

def _stretch_sample(np: Any, arr, mask=None):
    """§5.5 input sample ``S``: finite WB-only mapped values, excluding the
    exact display clip boundaries ``0.0`` / ``1.0`` but **keeping preserved
    headroom above ``1.0``** (PHI-AUTO-HISTOGRAM-UX-V1).

    RGB data is reduced to its Rec.601 luminance (a single global BP/WP pair
    applies to every channel); mono data is used directly.  A 2D validity mask
    (``mask[pixel] > 0``) is applied when provided.  The result is capped via
    the deterministic stride.

    Exclusion semantics: exact ``0.0`` (the sub-black display floor / legacy
    clipped black) and exact ``1.0`` (the legacy display-window top) stay
    excluded exactly as the ratified §5.5 step 1 requires, so in-window
    ``[0, 1]`` analysis buffers produce the *same* sample as before; every
    finite value **above** ``1.0`` (preserved analysis headroom) is now kept
    so the estimator can select a white point above ``1`` when a meaningful
    bright tail exists.  Non-finite values are always excluded.
    """
    lum = _luminance(np, arr)
    flat = lum.ravel()
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 2 and m.shape == lum.shape and m.size == flat.size:
            flat = flat[m.ravel() > 0]
    finite = flat[np.isfinite(flat)]
    finite = finite[(finite > 0.0) & (finite != 1.0)]
    if finite.size == 0:
        return None
    return _cap_sample(np, finite)


def compute_auto_stretch_float(mapped, mask=None, sep: float = ANCHOR_SEP) -> Tuple[float, float]:
    """Auto Stretch black/white points (§5.5 exact algorithm, analysis units).

    PHI-AUTO-HISTOGRAM-UX-V1: operates on the WB-only mapped float analysis
    buffer **in its preserved analysis units** — finite values above the
    legacy display-window top ``1.0`` (preserved HDR headroom) are part of the
    input sample (only the exact legacy clip boundaries ``0.0``/``1.0`` stay
    excluded, so in-window ``[0, 1]`` buffers are bit-identical to the
    ratified §5.5 algorithm), and the white point may therefore be selected
    **above ``1``** when a meaningful bright tail exists.  There is no hidden
    ``[0, 1]`` cap: the final separation clip is bounded by the *robust
    analysis high* ``D = max(1.0, p99.5 of the stretch sample)`` — the same
    robust top percentile as the histogram plot high — so a single extreme
    pixel can never pull the white point up to the outlier/max, while a
    genuinely dense bright population extends it.  (Legacy in-window data has
    ``p99.5(S) <= 1.0``, so ``D == 1.0`` and the clip is bit-identical to the
    ratified §5.5 spec.)
    Steps:

    1. keep finite pixels (and ``mask[pixel] > 0`` when a mask is given),
       excluding the exact clip boundaries ``0.0`` and ``1.0``;
    2. ``|S| < 20`` -> deterministic defaults ``(0.01, 0.99)``;
    3. ``p005 = percentile(S, 0.5)``, ``p60 = percentile(S, 60)``,
       ``p995 = percentile(S, 99.5)``;
    4. background ``B = { s <= p60 }``; ``bg = median(B)``;
       ``sigma = 1.4826 * MAD(B)``;
    5. ``bp = clip(max(p005, bg - 2.8 sigma), 0, D - sep)``;
    6. ``wp = clip(max(p995, bg + 8 sigma, bp + sep), bp + sep, D)`` where
       ``D = max(1.0, p99.5 of S)`` (legacy in-window data: ``D == 1.0`` and
       steps 5-6 are bit-identical to the ratified spec);
    7. degenerate fallback (empty/non-finite ``B`` or ``sigma == 0``):
       ``bp = p005``, ``wp = p995``, then clamp/separate as in 5-6.

    There is **no** min/max renormalization step.
    """
    np = _load_numpy()
    if np is None:
        return AUTO_STRETCH_DEFAULTS
    arr = np.asarray(mapped, dtype=np.float64)
    if arr.size == 0 or arr.ndim not in (2, 3):
        return AUTO_STRETCH_DEFAULTS

    S = _stretch_sample(np, arr, mask)
    if S is None or S.size < AUTO_STRETCH_MIN_SAMPLE:
        return AUTO_STRETCH_DEFAULTS

    p005 = float(np.percentile(S, AUTO_STRETCH_PCT_LO))
    p60 = float(np.percentile(S, AUTO_STRETCH_PCT_BG))
    p995 = float(np.percentile(S, AUTO_STRETCH_PCT_HI))

    B = S[S <= p60]
    if B.size == 0 or not np.all(np.isfinite(B)):
        bp, wp = p005, p995
    else:
        bg = float(np.median(B))
        mad = float(np.median(np.abs(B - bg)))
        sigma = AUTO_STRETCH_MAD_SCALE * mad
        if sigma == 0.0 or not np.isfinite(sigma):
            bp, wp = p005, p995
        else:
            bp = max(p005, bg - AUTO_STRETCH_BG_SPREAD_BP * sigma)
            wp = max(p995, bg + AUTO_STRETCH_BG_SPREAD_WP * sigma)

    # PHI-AUTO-HISTOGRAM-UX-V1 final separation/clip.  The legacy [0, 1]
    # display-window ceiling becomes the robust analysis high D = max(1.0,
    # p99.5 of the stretch sample): in-window data keeps D == 1.0 (bit-
    # identical legacy output); analysis buffers with meaningful headroom
    # extend D above 1 so WP > 1 is selectable, while an isolated extreme
    # outlier never raises D (the 99.5th percentile is outlier-robust).  The
    # ``bp + sep`` lower bound of the wp clip enforces the spec's
    # ``max(..., bp + sep)`` term exactly.
    D = max(1.0, float(np.percentile(S, X_RANGE_PCT_HI)))
    bp = float(np.clip(bp, 0.0, D - sep))
    wp = float(np.clip(wp, bp + sep, D))
    return (bp, wp)


# --------------------------------------------------------------------------
# §5.6 — Auto WB (true-background-band algorithm)
# --------------------------------------------------------------------------

def compute_auto_wb_float(mapped) -> Tuple[float, float, float]:
    """Auto WB gains (§5.6 true-background-band algorithm).

    Estimates from the *pristine pre-WB* mapped float analysis buffer (never
    from already-WB pixels), so repeated AutoWB is deterministic/idempotent.
    The ``(0, 0.98)`` per-channel exclusion below removes both the display
    window top and any preserved anchor headroom (values ``>= 0.98``), so the
    algorithm output is unchanged by PHI-R3 headroom preservation.

    1. common finite RGB set (>=3 channels, all three channels finite);
    2. every channel in ``(0, 0.98)`` (excludes zero/dark borders and
       near-saturated stars);
    3. true-background luminance band: keep ``p5 <= lum <= p60`` (degenerate
       ``p60 <= p5`` -> neutral);
    4. per-channel centre = median on the *same* selected pixels;
    5. gains relative green, clipped to ``[0.2, 5.0]``; any ``centre <= 1e-6``
       -> neutral;
    6. fewer than 64 valid pixels after exclusion -> neutral.

    Deterministic sampling cap applies before the percentile/median work.
    """
    np = _load_numpy()
    if np is None:
        return NEUTRAL_WB
    arr = np.asarray(mapped, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return NEUTRAL_WB

    R = arr[..., 0]
    G = arr[..., 1]
    B = arr[..., 2]
    valid = (
        np.isfinite(R)
        & np.isfinite(G)
        & np.isfinite(B)
        & (R > 0.0)
        & (G > 0.0)
        & (B > 0.0)
        & (R < AUTO_WB_SATURATION)
        & (G < AUTO_WB_SATURATION)
        & (B < AUTO_WB_SATURATION)
    )

    idx = np.flatnonzero(valid.ravel())
    if idx.size > MAX_SAMPLE_PIXELS:
        stride = max(1, int(-(-idx.size // MAX_SAMPLE_PIXELS)))
        idx = idx[::stride]
    if idx.size < AUTO_WB_MIN_SAMPLE:
        return NEUTRAL_WB

    Rv = R.ravel()[idx]
    Gv = G.ravel()[idx]
    Bv = B.ravel()[idx]
    lum = AUTO_WB_LUMA[0] * Rv + AUTO_WB_LUMA[1] * Gv + AUTO_WB_LUMA[2] * Bv

    p5 = float(np.percentile(lum, 5.0))
    p60 = float(np.percentile(lum, 60.0))
    if not (p60 > p5):
        return NEUTRAL_WB

    band = (lum >= p5) & (lum <= p60)
    sel = idx[band]
    if sel.size < AUTO_WB_MIN_SAMPLE:
        return NEUTRAL_WB

    centre_r = float(np.median(R.ravel()[sel]))
    centre_g = float(np.median(G.ravel()[sel]))
    centre_b = float(np.median(B.ravel()[sel]))
    if min(centre_r, centre_g, centre_b) <= AUTO_WB_CENTRE_FLOOR:
        return NEUTRAL_WB

    gain_r = float(np.clip(centre_g / centre_r, AUTO_WB_GAIN_MIN, AUTO_WB_GAIN_MAX))
    gain_g = 1.0
    gain_b = float(np.clip(centre_g / centre_b, AUTO_WB_GAIN_MIN, AUTO_WB_GAIN_MAX))
    return (gain_r, gain_g, gain_b)
