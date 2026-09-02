"""Qt interactive histogram view (M14 / ZSSS-OTPUX-HIST-H1) — display-only.

Replaces the former static ``QLabel`` histogram pixmap with a real ``QWidget``
that reproduces the Tk ``HistogramWidget`` *interactions* (auto-zoom, reset
view, zoom, reset zoom and BP/WP line dragging) without importing Tk, the
scientific engine, or any plotting/image library.

Since H1 the widget has two data inputs:

* ``set_model`` — the *authoritative* immutable result of
  :func:`seestar.gui_qt.preview_analysis.compute_histogram_float`: a 512-bin
  float-domain model with explicit, distinct roles (PHI-AUTO-HISTOGRAM-UX-V1):
  the preserved **analysis/control domain** (``range``/``full_range`` =
  ``(0, upper)``, ``upper = max(1.0, finite max)`` — the stats/BP-WP marker
  truth), the **bin/plot range** ``bin_range = (0, bin_hi)`` the 512 bins
  actually live in (robust plot high, so a sparse extreme finite tail can
  never stretch the bins into a few widely spaced spikes), per-channel
  ``counts``/``log_counts``/``stats`` on the exact same deterministic sample,
  a robust plotted X range and per-channel ``overflow`` counts (in-domain
  values above the plotted bin high — never silently dropped: their extent
  stays visible through ``full_range`` and the stats ``max``).  This is what
  production Option-A previews feed; no ``QImage`` round-trip is involved for
  the histogram or its statistics.  The histogram is **analysis data** (the
  WB-only float buffer): bins/stats above the display white level ``1.0`` are
  preserved HDR headroom and are never conflated with the bounded ``uint8``
  display histogram (the legacy ``set_data`` path).
* ``set_data`` — the legacy single-array compatibility path, which still takes
  a WB-only ``QImage`` and computes the historical 256-bin ``uint8`` histogram
  via :mod:`seestar.gui_qt.preview_adjust`.  It is retained only so old
  producers keep working; the Option-A model is authoritative.

Rendering draws the model bars from ``log_counts`` (log-space heights) so fine

tonal detail stays readable, overlays the R/G/B (or L) channels on the same
axes, labels the plotted X domain, and draws the BP/WP lines in the **current
analysis units** shared with the stretch sliders (PHI-R3.1: markers and drags
operate over ``[0, marker_upper]`` with ``marker_upper = max(1.0, finite max)``
— the grid-ceiling control domain the owning MainWindow pushes via
:meth:`set_analysis_domain` — so a white point above ``1`` is a first-class
marker position; the legacy QImage path keeps the historical ``[0, 1]``
display-level window).  Model bars are placed in the model's **bin range**
(PHI-AUTO-HISTOGRAM-UX-V1), which is at most the analysis range and equals it
whenever the top is dense or the sample small: the plotted curve therefore
stays dense inside the auto/default visible windows even when a sparse extreme
tail exists, the overflow is drawn as an explicit plot-top marker with the
count available as :attr:`overflow_total`, and auto/manual zoom uses the robust
plotted range inside the binned domain.  Since FRP-H1 the model also carries a
**full-domain histogram** (``full_log_counts`` over ``full_hist_range`` == the
full analysis range ``(0, upper)`` — the complete sampled distribution, tail
included), and the view owns an explicit **persistent view mode**
(``_view_mode``: ``"default"`` / ``"auto"`` / ``"full"`` / ``"manual"``):
``reset_histogram_view`` / ``reset_zoom`` select the FULL mode, which shows the
real full-domain bars over the true analysis maximum *and persists across new
models* (a new model keeps FULL semantics: view + bars follow the new model's
full domain), while the legacy ``auto_zoom_enabled`` bool and ``_frozen_range``
remain the public/back-compat surface.  The *analysis/control* extent stays the
marker domain and the stats truth.  BP/WP line dragging emits ``rangeChanged``
**live/coalesced** at ~25 ms during the drag and an exact final emission on
release (the Qt equivalent of Tk's ``update_stretch_from_histogram``).

The widget never mutates any image: it only holds a computed model/histogram
and a percentile level.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .preview_adjust import (
    BP_WP_MIN_SEPARATION,
    compute_histogram,
    compute_histogram_percentile,
    normalize_bp_wp,
    quantize_bp_wp,
)

# Channel-name -> bar colour (matches the earlier M10 display-histogram
# palette so the surface stays visually consistent).
_CHANNEL_COLORS: Dict[str, QColor] = {
    "L": QColor(225, 225, 225),
    "R": QColor(225, 70, 70),
    "G": QColor(70, 200, 70),
    "B": QColor(70, 120, 225),
}

_BACKGROUND = QColor(22, 22, 24)
_BLACK_POINT_COLOR = QColor(255, 170, 170)
_WHITE_POINT_COLOR = QColor(170, 170, 255)
_EMPTY_TEXT_COLOR = QColor(120, 120, 120)
_AXIS_TEXT_COLOR = QColor(150, 150, 150)
# Plot-top / overflow boundary marker (PHI-AUTO-HISTOGRAM-UX-V1): a dashed
# vertical line drawn at the bin-range high whenever in-domain values exist
# above the plotted bins.
_OVERFLOW_MARKER_COLOR = QColor(255, 200, 90)

# Default black/white line positions (Tk stretch defaults).
_DEFAULT_BLACK_POINT = 0.01
_DEFAULT_WHITE_POINT = 0.99
# Single authoritative BP/WP minimum separation (one slider/spin step), shared
# with the MainWindow stretch controls via ``preview_adjust``.
_MIN_SEPARATION = BP_WP_MIN_SEPARATION

# Live-drag coalescing cadence (ZSSS-OTPUX §5.7): intermediate BP/WP emissions
# during a drag are timer-coalesced to roughly this interval; the release
# emission is never dropped.
_LIVE_DRAG_INTERVAL_MS = 25

# Bottom strip reserved for the plotted X-domain labels.
_AXIS_MARGIN_BOTTOM = 14.0

# Deterministic minimum zoom width for the robust plotted X range (matches the
# legacy ``max(0.02, p99.5)`` lower bound so a degenerate-but-valid robust
# range never collapses to a sliver).
_MIN_ZOOM_WIDTH = 0.02


def _validated_model_range(model_range) -> tuple:
    """Return a validated analysis range ``(lo, hi)`` from model metadata.

    Accepts a finite ``(lo, hi)`` pair with ``0 <= lo < hi``; everything else
    falls back to the display-level window ``(0.0, 1.0)``.  The model range is
    the *preserved analysis domain* ``(0, upper)`` with
    ``upper = max(1.0, finite max)`` produced by ``compute_histogram_float``
    (PHI-R3), so ``hi`` may legitimately exceed ``1.0`` when the analysis
    buffer carries HDR headroom.
    """
    if not isinstance(model_range, (tuple, list)) or len(model_range) != 2:
        return (0.0, 1.0)
    try:
        lo = float(model_range[0])
        hi = float(model_range[1])
    except (TypeError, ValueError):
        return (0.0, 1.0)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return (0.0, 1.0)
    if not (0.0 <= lo < hi):
        return (0.0, 1.0)
    return (lo, hi)


def _validated_x_range(x_range, upper: float = 1.0) -> tuple:
    """Return a validated robust plotted X range ``(lo, hi)`` or the full range.

    Accepts the model ``x_range`` metadata; rejects non-sequence values,
    non-finite bounds, out-of-domain bounds and degenerate ``hi <= lo`` ranges
    by falling back to the explicit full ``(0, upper)`` analysis range.  A
    valid range that is narrower than :data:`_MIN_ZOOM_WIDTH` is widened
    deterministically (centred, clamped to ``[0, upper]``) so zooming never
    collapses to a sliver.
    """
    if not isinstance(x_range, (tuple, list)) or len(x_range) != 2:
        return (0.0, upper)
    try:
        lo = float(x_range[0])
        hi = float(x_range[1])
    except (TypeError, ValueError):
        return (0.0, upper)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return (0.0, upper)
    if not (0.0 <= lo < hi <= upper):
        return (0.0, upper)
    if hi - lo < _MIN_ZOOM_WIDTH:
        mid = 0.5 * (lo + hi)
        lo = mid - 0.5 * _MIN_ZOOM_WIDTH
        hi = mid + 0.5 * _MIN_ZOOM_WIDTH
        if lo < 0.0:
            hi += -lo
            lo = 0.0
        if hi > upper:
            lo -= hi - upper
            hi = upper
        lo = max(0.0, lo)
        hi = min(upper, hi)
    return (lo, hi)


def _reconcile_range_to_upper(frozen, upper: float) -> tuple:
    """Revalidate a frozen/manual view range against the current data domain.

    PHI-R3.2 (F2): when the model analysis domain changes (a new preview / WB
    derivation with a different ``upper``), a previously frozen manual zoom
    window may no longer fit the new domain (e.g. a zoom to ``(0.1, 3.98)``
    followed by a model whose range ends at ``1.2``).  A window that still
    fits is preserved verbatim (a valid manual range is kept whenever
    possible); one whose bounds exceed the new domain is clamped into it;
    only a window that would become degenerate after clamping (``hi - lo``
    below the deterministic minimum width) falls back to the full analysis
    range ``(0, upper)``.  Deterministic, never inverted, never beyond the
    data domain.
    """
    if not isinstance(frozen, (tuple, list)) or len(frozen) != 2:
        return (0.0, upper)
    try:
        lo = float(frozen[0])
        hi = float(frozen[1])
    except (TypeError, ValueError):
        return (0.0, upper)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return (0.0, upper)
    lo = min(max(lo, 0.0), upper)
    hi = min(max(hi, 0.0), upper)
    if hi - lo < _MIN_ZOOM_WIDTH:
        return (0.0, upper)
    return (lo, hi)


def format_histogram_stats(stats: Optional[Dict[str, Dict[str, float]]]) -> Optional[str]:
    """Return a deterministic per-channel stats summary (analysis domain).

    Each channel reports ``min``/``max``/``median``/``mean``/``std`` (the five
    ratified stats), labelled ``R``/``G``/``B`` (or ``L`` for mono), joined by
    "·".  The values are the preserved-analysis-domain stats of the float
    model (PHI-R3): with HDR headroom the ``max`` may legitimately exceed
    ``1.0`` (analysis headroom above the display window) — this is analysis
    data, never the bounded uint8 display histogram.  Returns ``None`` when
    ``stats`` is empty/``None``.
    """
    if not stats:
        return None
    parts = []
    for name in ("R", "G", "B", "L"):
        s = stats.get(name)
        if s is None:
            continue
        parts.append(
            f"{name} {s['min']:.3f}–{s['max']:.3f} "
            f"med {s['median']:.3f} mean {s['mean']:.3f} std {s['std']:.3f}"
        )
    if not parts:
        return None
    return " · ".join(parts)


def format_histogram_overflow(model: Optional[Dict[str, Any]]) -> str:
    """Deterministic overflow annotation for the float model status line.

    PHI-AUTO-HISTOGRAM-UX-V1: when the model reports in-domain values above
    the plotted bin high (``overflow_total > 0``), a compact truthful suffix
    exposes them (``+N px above plot top``), so the sparse extreme tail is
    visible on the UI in *every* zoom state — never silently dropped.  Returns
    an empty string when the model is absent or has no overflow (legacy QImage
    histograms and no-tail float models keep their exact previous text).
    """
    if not model:
        return ""
    total = int(model.get("overflow_total") or 0)
    if total <= 0:
        return ""
    bin_range = model.get("bin_range")
    top = ""
    if isinstance(bin_range, (tuple, list)) and len(bin_range) == 2:
        try:
            top = f" at {float(bin_range[1]):.3g}"
        except (TypeError, ValueError):
            top = ""
    return f" (+{total} px above plot top{top})"


class HistogramView(QWidget):
    """A small interactive per-channel histogram with BP/WP lines and zoom.

    ``set_model`` feeds the authoritative float-domain model (512 bins, RGB
    overlay or L, log-space heights, exact stats); ``set_data`` is the legacy
    QImage compatibility path.  ``set_range`` positions the black/white lines;
    ``zoom_histogram`` / ``reset_histogram_view`` / ``reset_zoom`` reproduce
    the Tk zoom behaviours; dragging a line emits ``rangeChanged`` live
    (coalesced ~25 ms) and exactly once on release.
    """

    # Emitted with the current (0-1) black/white points: live/coalesced during
    # a drag and exactly once (final, authoritative) on release.
    rangeChanged = Signal(float, float)
    # The inline compact view uses this as an obvious double-click expansion
    # seam.  Detached views may ignore it; emitting never computes data.
    expandRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Authoritative float model (compute_histogram_float result) or None.
        self._model: Optional[Dict[str, Any]] = None
        # Validated analysis range ``(lo, hi)`` of the current float model
        # (``hi = max(1.0, finite max)`` — PHI-R3).  Falls back to the
        # display-level window ``(0, 1)`` for legacy/absent metadata.
        self._model_range: tuple = (0.0, 1.0)
        # Validated plotting/bin range ``(lo, hi)`` where the 512 model bars
        # actually live (PHI-AUTO-HISTOGRAM-UX-V1): ``bin_range`` from the
        # model when present (``<= model range``; equal to it when no sparse
        # extreme tail exists), falling back to the model range for
        # synthetic/legacy models without the key.
        self._bin_range: tuple = (0.0, 1.0)
        # Total number of in-domain values above the plotted bin high
        # (``overflow_total`` model metadata; 0 for no-overflow / legacy).
        self._overflow_total: int = 0
        # BP/WP marker domain upper bound (PHI-R3.1).  The black/white point
        # markers and the drag conversion operate in the current analysis
        # units: ``[0, _marker_upper]`` with
        # ``_marker_upper = max(1.0, finite max)`` for Option-A float models,
        # ``1.0`` for the legacy [0, 1] QImage path.  The owning MainWindow is
        # authoritative and pushes the synchronous domain via
        # :meth:`set_analysis_domain`; ``set_model``/``set_data`` adopt the
        # matching value from their data source so the two always agree.
        self._marker_upper: float = 1.0
        # Legacy 256-bin counts dict (``{channel: int64 array}``) or None.
        self._histogram: Optional[Dict[str, Any]] = None
        self._percentile_99_5: float = 1.0
        # Validated robust plotted X range from the float model (``(lo, hi)``)
        # or ``None`` while on the legacy path (which zooms ``[0, p99.5]``).
        self._x_range: Optional[tuple] = None
        self._black_point: float = _DEFAULT_BLACK_POINT
        self._white_point: float = _DEFAULT_WHITE_POINT
        self._view_min: float = 0.0
        self._view_max: float = 1.0
        # Manual-zoom window ``(view_min, view_max)`` preserved across data
        # refreshes (Tk ``freeze_x_range`` / ``_stored_xlim`` semantics), or
        # ``None`` while the view tracks the full data range.  Cleared by
        # ``reset_histogram_view`` / ``reset_zoom`` and when data is cleared.
        self._frozen_range: Optional[tuple] = None
        # Explicit persistent view semantics (FRP-H1): ``"default"``
        # (display-level window ``(0, 1)``), ``"auto"`` (robust zoom window,
        # never frozen), ``"full"`` (complete full-domain distribution over the
        # true analysis maximum — persistent across new models) or
        # ``"manual"`` (frozen zoom window).  ``auto_zoom_enabled`` (bool) and
        # ``_frozen_range`` stay the public/back-compat surface, but behaviour
        # is driven from this mode.
        self._view_mode: str = "default"
        self.auto_zoom_enabled: bool = False
        self._drag_line: Optional[str] = None
        # Live-drag coalescing (single-shot, ~25 ms).  A pending intermediate
        # emission is dropped/replaced; the release emit is never dropped.
        self._live_drag_timer = QTimer(self)
        self._live_drag_timer.setSingleShot(True)
        self._live_drag_timer.setInterval(_LIVE_DRAG_INTERVAL_MS)
        self._live_drag_timer.timeout.connect(self._emit_live_drag)
        self.setMinimumSize(256, 64)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ data
    def set_model(self, model: Optional[Dict[str, Any]]) -> None:
        """Feed the authoritative float-domain histogram model.

        ``model`` is the immutable result of
        :func:`~seestar.gui_qt.preview_analysis.compute_histogram_float`
        (``bins``/``range``/``channels``/``counts``/``log_counts``/``stats``/
        ``x_range``/``full_range``/``bin_range``/``overflow_total`` and, since
        FRP-H1, ``full_counts``/``full_log_counts``/``full_hist_range``).  The
        model ``range`` is the preserved analysis/control domain ``(0, upper)``
        (PHI-R3): zoom bounds, marker-domain fallback and the F2 frozen-range
        reconcile operate against it, and ``zoom_histogram`` validates the
        robust X range against its upper bound.  The model ``bin_range`` is
        where the 512 plotted bins actually live (PHI-AUTO-HISTOGRAM-UX-V1):
        bars are placed in that domain, so a sparse extreme tail never
        stretches them into a few widely spaced spikes.  ``overflow_total``
        counts the in-domain values above the plotted bin high (their full
        extent stays truthful in ``full_range`` and the per-channel stats
        ``max``).  FRP-H1: when the view is in FULL mode (a previous
        ``reset_histogram_view`` / ``reset_zoom``), a new model keeps FULL
        semantics — the view follows the new model's full domain and paints the
        new model's complete full-domain distribution
        (``full_log_counts``/``full_hist_range``), so the user's "show the full
        distribution" choice persists instead of silently falling back to the
        auto/default window.  When the analysis domain changes, any
        frozen/manual view range is revalidated against the new upper
        (PHI-R3.2): a still-valid manual range is preserved verbatim, an
        out-of-domain one is clamped into the new domain, and only a degenerate
        result falls back to the full range — the painted axis never extends
        beyond the data domain and the inline/detached surfaces reconcile
        identically (they receive the same model).  Passing the same model
        object again only repaints (cheap), so a refresh that merely moves the
        BP/WP markers does not reset a manual zoom.
        """
        if model is not None and model is self._model:
            # Same model object: cheap repaint (no frozen-zoom reset).  A
            # ``None`` model always falls through so ``clear()`` also drops any
            # legacy histogram left behind by a prior ``set_data``.
            self.update()
            return
        self._model = model
        self._histogram = None
        if model is None:
            self._percentile_99_5 = 1.0
            self._x_range = None
            self._frozen_range = None
            self._view_mode = "default"
            self._model_range = (0.0, 1.0)
            self._bin_range = (0.0, 1.0)
            self._overflow_total = 0
            self._marker_upper = 1.0
        else:
            self._model_range = _validated_model_range(model.get("range"))
            # PHI-AUTO-HISTOGRAM-UX-V1: the plotted bars live in the model's
            # explicit bin range (fallback: the analysis range for models
            # without the key — synthetic/legacy-compatible).
            self._bin_range = _validated_model_range(
                model.get("bin_range") or model.get("range")
            )
            self._overflow_total = int(
                model.get("overflow_total") or 0
            )
            x_range = model.get("x_range")
            self._x_range = _validated_x_range(x_range, self._model_range[1])
            self._percentile_99_5 = self._x_range[1]
            # The BP/WP marker domain is NOT re-derived from the model range:
            # the owning MainWindow is authoritative and pushes the synchronous
            # grid-ceiling control domain via set_analysis_domain() (which is
            # >= the model's raw upper by construction).  Keeping it here would
            # shrink the domain below the control grid ceiling on every async
            # model application.
            # PHI-R3.2 (F2): a frozen/manual view window from a previous model
            # (possibly wider than the new domain) must be revalidated against
            # the new upper *before* ``_apply_view_after_data`` restores it, so
            # a stale window can never survive a domain shrink (and a valid one
            # survives a shrink/grow unchanged).
            if self._frozen_range is not None:
                reconciled = _reconcile_range_to_upper(
                    self._frozen_range, self._model_range[1]
                )
                self._frozen_range = reconciled
                self._view_min, self._view_max = reconciled
        self._apply_view_after_data()
        self.update()

    def set_data(self, image: Optional[QImage]) -> None:
        """Legacy compatibility path: feed a WB-only display ``QImage``.

        Keeps the historical 256-bin ``uint8`` histogram for old single-array
        producers (display-level domain ``[0, 1]``, markers included).
        Production Option-A previews must use :meth:`set_model`.

        PHI-R3.3 (F3): a model→legacy transition clears any frozen/manual
        view state left over from the float model and restores a valid legacy
        ``[0, 1]`` window automatically (via :meth:`_apply_view_after_data`),
        so legacy bars are never painted on an axis left above ``1``.  A
        legacy→legacy data refresh (no float model before) keeps the legacy
        manual-zoom semantics unchanged (a frozen window is preserved).
        """
        had_model = self._model is not None
        self._model = None
        self._marker_upper = 1.0
        self._bin_range = (0.0, 1.0)
        self._overflow_total = 0
        if had_model:
            # FRP-H1 / PHI-R3.3 (F3): a model→legacy transition drops the
            # float view policy (frozen window AND explicit mode) — the legacy
            # axis always corresponds to the legacy [0, 1] data, and a later
            # float model starts from the clean default window again.
            self._frozen_range = None
            self._view_mode = "default"
        if image is None or image.isNull():
            self._histogram = None
            self._percentile_99_5 = 1.0
            self._x_range = None
            self._frozen_range = None
        else:
            self._histogram = compute_histogram(image, bins=256)
            p99_5 = compute_histogram_percentile(image, 99.5)
            self._percentile_99_5 = 1.0 if p99_5 is None else p99_5
            self._x_range = None
        self._apply_view_after_data()
        self.update()

    def set_legacy_data(self, histogram: Optional[Dict[str, Any]], percentile_99_5: float = 1.0) -> None:
        """Inject a pre-computed legacy 256-bin histogram + percentile (no recompute).

        Used by the legacy path's cache: ``MainWindow`` computes the histogram
        and percentile once per ``(source, WB)`` revision and feeds them here on
        refresh without a ``QImage`` round-trip recompute.  Mirrors
        :meth:`set_data` but skips ``compute_histogram`` /
        ``compute_histogram_percentile`` entirely.

        PHI-R3.3 (F3): a model→legacy transition clears any frozen/manual
        view state left over from the float model and restores a valid legacy
        ``[0, 1]`` window automatically, so the legacy axis always corresponds
        to the legacy ``[0, 1]`` data without a manual reset.  A legacy→legacy
        data refresh (no float model before) keeps the legacy manual-zoom
        semantics unchanged.
        """
        had_model = self._model is not None
        self._model = None
        self._marker_upper = 1.0
        self._bin_range = (0.0, 1.0)
        self._overflow_total = 0
        if had_model:
            # FRP-H1 / PHI-R3.3 (F3): see :meth:`set_data`.
            self._frozen_range = None
            self._view_mode = "default"
        self._histogram = histogram
        self._percentile_99_5 = float(percentile_99_5)
        self._x_range = None
        self._apply_view_after_data()
        self.update()

    def set_analysis_domain(self, upper: float) -> None:
        """Push the synchronous BP/WP marker domain upper (PHI-R3.1).

        The owning MainWindow calls this whenever the Option-A analysis buffer
        changes so the black/white markers and the drag conversion operate in
        the current analysis units ``[0, upper]`` (``upper = max(1.0, finite
        max)``) *before* the async histogram model lands; the model applies
        the identical value on arrival.  Values below ``1.0`` or non-finite
        input are ignored (the marker domain never shrinks below the legacy
        display window).  Current markers are re-normalized into the new
        domain without inversion.
        """
        try:
            upper = float(upper)
        except (TypeError, ValueError):
            return
        if not math.isfinite(upper) or upper < 1.0:
            return
        if abs(upper - self._marker_upper) < 1e-12:
            return
        self._marker_upper = upper
        self._black_point, self._white_point = normalize_bp_wp(
            self._black_point,
            self._white_point,
            max_value=self._marker_upper,
        )
        self.update()

    def _apply_view_after_data(self) -> None:
        """Re-apply the persistent view mode after a data (model) change.

        FRP-H1 view-mode dispatch (drive behaviour from ``_view_mode``, with
        the back-compat ``auto_zoom_enabled`` bool taking priority so the
        legacy wiring keeps working):

        * ``auto`` (``auto_zoom_enabled``) — zoom to the robust plotted window
          of the current data, never frozen;
        * ``full`` — view = the full-domain window of the current data (the
          true analysis maximum for a float model, ``[0, 1]`` for legacy), so
          a user Reset/Full choice persists across new models;
        * ``manual`` — preserve the frozen window (already reconciled to the
          current data domain via ``_reconcile_range_to_upper`` on shrink);
        * ``default`` (and the legacy path) — the display-level ``(0, 1)``
          window (BP/WP live there).  When no headroom exists the display
          window equals the full analysis range, so this is identical to the
          pre-PHI-R3 behaviour.
        """
        if not self.has_data:
            # Empty surface (``set_model(None)`` / ``set_data(None)`` /
            # ``set_legacy_data(None)`` clears): report the clean default
            # window, never a stale zoom window from before the clear.
            self._view_min = 0.0
            self._view_max = 1.0
            return
        if self.auto_zoom_enabled:
            self._view_mode = "auto"
            self._view_min, self._view_max = self._zoom_window()
        elif self._view_mode == "full":
            self._view_min, self._view_max = self._full_range()
        elif self._view_mode == "manual" and self._frozen_range is not None:
            # Preserve a manual zoom across the refresh (Tk freeze_x_range).
            self._view_min, self._view_max = self._frozen_range
        else:
            self._view_mode = "default"
            self._view_min = 0.0
            self._view_max = 1.0

    def clear(self) -> None:
        """Clear the histogram surface (no preview)."""
        self.set_model(None)

    @property
    def has_data(self) -> bool:
        """True while a model or legacy histogram is present."""
        return self._model is not None or self._histogram is not None

    @property
    def model(self) -> Optional[Dict[str, Any]]:
        """The authoritative float-domain model, or ``None``."""
        return self._model

    @property
    def histogram(self) -> Optional[Dict[str, Any]]:
        """Per-channel bin counts ``{channel: int64 array}`` (model or legacy)."""
        if self._model is not None:
            return self._model.get("counts")
        return self._histogram

    @property
    def stats(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Per-channel ``{min, max, median, mean, std}`` stats, or ``None``."""
        if self._model is None:
            return None
        return self._model.get("stats")

    @property
    def bin_range(self) -> tuple:
        """Validated plotting/bin range the model bars live in (or ``(0, 1)``).

        PHI-AUTO-HISTOGRAM-UX-V1: at most the analysis range and equal to it
        when no sparse extreme tail exists.  ``None``-model and legacy views
        report the display-level window ``(0, 1)``.
        """
        return self._bin_range

    @property
    def overflow_total(self) -> int:
        """In-domain values above the plotted bin high (model truth, or 0).

        PHI-AUTO-HISTOGRAM-UX-V1: values the sparse extreme tail places above
        the plotting/bin range — never silently dropped (their extent stays
        visible in the per-channel stats ``max`` and ``full_range``).
        """
        return self._overflow_total

    # -------------------------------------------------------------- BP/WP
    @property
    def black_point(self) -> float:
        """Current black-point line position in the 0-1 level scale."""
        return self._black_point

    @property
    def white_point(self) -> float:
        """Current white-point line position in the 0-1 level scale."""
        return self._white_point

    def set_range(self, bp: float, wp: float) -> None:
        """Position the black/white lines (analysis-unit values, Tk ``set_range``).

        Enforces ``0 <= BP < WP <= marker_upper`` with the shared deterministic
        minimum separation via :func:`preview_adjust.normalize_bp_wp` (the same
        seam used by the MainWindow stretch controls, so the handles always
        agree with the slider/spin state).  ``marker_upper`` is the current
        analysis domain upper (``max(1.0, finite max)``, PHI-R3.1) for
        Option-A float models and ``1.0`` for the legacy QImage path, so a
        white point above ``1`` is a first-class marker position.  Non-finite
        inputs fall back to the neutral defaults deterministically.  A no-op
        while a drag is active so a live-drag echo from the sliders never
        clobbers the authoritative in-flight line position (avoids jitter /
        rounding feedback).
        """
        if self._drag_line is not None:
            return
        self._black_point, self._white_point = normalize_bp_wp(
            bp, wp, max_value=self._marker_upper
        )
        self.update()

    # ----------------------------------------------------------- zoom/view
    @property
    def view_range(self) -> tuple:
        """Current zoom window ``(view_min, view_max)`` in 0-1 level space."""
        return (self._view_min, self._view_max)

    def zoom_histogram(self) -> None:
        """Zoom the X axis to the authoritative data range (Tk ``zoom_histogram``).

        The Option-A float model zooms to its validated robust plotted X range
        (both ``lo`` and ``hi``; the robust high end can exceed ``1.0`` when
        the analysis buffer carries dense HDR headroom); the legacy path keeps
        the historical ``[0, max(0.02, p99.5)]`` window.  Invalid/degenerate
        model metadata already fell back to the full analysis range in
        :meth:`set_model`.

        FRP-H1 view semantics: with auto-zoom enabled the view enters AUTO
        mode (robust zoom window, never frozen); otherwise it enters MANUAL
        mode and the zoom window is frozen (``_frozen_range``) so it survives
        data refreshes (Tk stores ``_stored_xlim`` only while ``freeze_x_range``
        is set).
        """
        self._view_min, self._view_max = self._zoom_window()
        if self.auto_zoom_enabled:
            # AUTO: unfrozen robust zoom (do not freeze across refreshes).
            self._view_mode = "auto"
        else:
            # MANUAL: a manual zoom is frozen across refreshes.
            self._view_mode = "manual"
            self._frozen_range = (self._view_min, self._view_max)
        self.update()

    def _zoom_window(self) -> tuple:
        """Return the zoom window ``(view_min, view_max)`` for the current data.

        Model data uses the validated robust ``x_range``; legacy data uses the
        historical ``[0, max(0.02, p99.5)]`` window.
        """
        if self._x_range is not None:
            return self._x_range
        return (0.0, max(0.02, self._percentile_99_5))

    def _full_range(self) -> tuple:
        """Full X window for the current data (full analysis or ``[0, 1]``).

        Float model: the explicit full **analysis** range ``(0, upper)``
        declared by the model (PHI-R3) — identical to the display window
        ``[0, 1]`` when no headroom exists.  FRP-H1: this is also the domain
        the full-domain bars live in (``full_hist_range`` == ``range`` == the
        true sampled maximum), so the FULL-mode view window always matches the
        painted full distribution.  Legacy data: the historical ``[0, 1]``
        display-level window.
        """
        if self._model is not None:
            return self._model_range
        return (0.0, 1.0)

    def reset_histogram_view(self) -> None:
        """Reset the X axis to the FULL distribution (Tk ``reset_histogram_view``).

        FRP-H1: enters the persistent FULL view mode — the view window is set
        to the complete full-domain range (``full_hist_range`` hi, i.e. the
        true analysis maximum ``_model_range[1]``) and the surface paints the
        model's **full-domain bars** (``full_log_counts`` over the full
        domain, no overflow marker): Reset/Full genuinely displays the whole
        sampled distribution, tail included, not just a widened axis around
        the robust-only bars.  The FULL mode is *persistent*: each subsequent
        ``set_model`` with a new model keeps FULL semantics (view + bars follow
        the new model's full domain) instead of silently falling back to the
        auto/default window.  Any frozen/manual window is cleared.  The legacy
        path returns to ``[0, 1]`` as before (its full domain).
        """
        self._view_mode = "full"
        self._frozen_range = None
        self._view_min, self._view_max = self._full_range()
        self.update()

    def reset_zoom(self) -> None:
        """Reset the X axis to the FULL distribution (Tk ``reset_zoom``).

        Identical to :meth:`reset_histogram_view`: enters the persistent FULL
        view mode (full-domain window + full-domain bars; see there).
        """
        self.reset_histogram_view()

    def set_view_range(self, view_min: float, view_max: float) -> None:
        """Apply a view-window snapshot supplied by the owning controller.

        PHI-R3.3 (F2): a coordinate snapshot must NOT manufacture a frozen
        (manual-zoom) state — this only sets the current view window and
        leaves the frozen-vs-unfrozen policy untouched.  Surfaces that must
        share the full policy (window + frozen state + auto-zoom) use
        :meth:`mirror_state_from`.  Never changes the histogram model, the
        markers or the analysis.
        """
        upper = self._model_range[1] if self._model is not None else 1.0
        validated = _validated_x_range((view_min, view_max), upper)
        self._view_min, self._view_max = validated
        self.update()

    def mirror_state_from(self, other) -> None:
        """Copy the authoritative inline view policy onto this surface (mirror).

        PHI-R3.3 (F2): MainWindow treats the inline ``HistogramView`` as the
        single owner of the view policy and calls this on the detached surface
        whenever it is (re)synchronized and after every model application, so
        the two surfaces can never diverge (in particular after a model
        analysis-domain shrink/grow).  Copies:

        * the auto-zoom flag;
        * the explicit view mode (``_view_mode`` — FRP-H1: ``"default"`` /
          ``"auto"`` / ``"full"`` / ``"manual"``), so a persistent FULL
          choice made on the inline surface stays FULL on the detached surface
          (and both follow the next model's full domain in lockstep);
        * the frozen-vs-unfrozen state: a genuine manual/robust zoom on the
          inline view (``other._frozen_range``) is copied verbatim (validated
          against this surface's model domain, which is the same model); when
          the inline view is **unfrozen**, this surface's frozen state is
          cleared too and only the current view-window coordinates are snapped
          — a plain snapshot, never a manufactured freeze;
        * the current view window.

        Requires both surfaces to hold the same model object (MainWindow
        invariant); never changes the model, the histogram data, the markers
        or the analysis.
        """
        self.auto_zoom_enabled = bool(getattr(other, "auto_zoom_enabled", False))
        self._view_mode = str(getattr(other, "_view_mode", "default"))
        upper = self._model_range[1] if self._model is not None else 1.0
        other_upper = (
            other._model_range[1] if other._model is not None else 1.0
        )
        limit = max(upper, other_upper)  # same model -> equal (defensive)
        other_frozen = getattr(other, "_frozen_range", None)
        if other_frozen is not None:
            validated = _validated_x_range(other_frozen, limit)
            self._frozen_range = validated
            self._view_min, self._view_max = validated
        else:
            self._frozen_range = None
            lo = float(getattr(other, "_view_min", 0.0))
            hi = float(getattr(other, "_view_max", 1.0))
            self._view_min, self._view_max = _validated_x_range((lo, hi), limit)
        self.update()

    # -------------------------------------------------------- drag plumbing
    # The mouse handlers below are thin wrappers around these small methods so
    # the interaction logic is unit-testable without synthesising mouse events.
    def _plot_rect(self) -> QRectF:
        margin = 4.0
        return QRectF(
            margin,
            margin,
            max(1.0, self.width() - 2.0 * margin),
            max(1.0, self.height() - margin - _AXIS_MARGIN_BOTTOM),
        )

    def _level_to_x(self, level: float) -> float:
        rect = self._plot_rect()
        span = self._view_max - self._view_min
        if span <= 0.0:
            return rect.left()
        return rect.left() + (level - self._view_min) / span * rect.width()

    def _x_to_level(self, x: float) -> float:
        rect = self._plot_rect()
        if rect.width() <= 0.0:
            return 0.0
        frac = (x - rect.left()) / rect.width()
        return self._view_min + frac * (self._view_max - self._view_min)

    def _start_drag_at(self, x: float) -> Optional[str]:
        """Pick the nearest line to ``x`` (pixel) and start dragging it."""
        if not self.has_data:
            return None
        pick = max(5.0, 0.02 * self.width())
        d_bp = abs(x - self._level_to_x(self._black_point))
        d_wp = abs(x - self._level_to_x(self._white_point))
        if d_bp <= pick and d_bp <= d_wp:
            self._drag_line = "min"
        elif d_wp <= pick:
            self._drag_line = "max"
        else:
            self._drag_line = None
        return self._drag_line

    def _drag_at(self, x: float) -> None:
        """Move the active line to pixel ``x`` (clamped + kept separated).

        Schedules a coalesced live ``rangeChanged`` emission (the intermediate
        emission is timer-coalesced; raw mouse events never each emit).  The
        in-flight handle position is snapped to the shared control-resolution
        grid *before* it is stored/emitted, so the handle, the stretch
        sliders/spins and the authoritative MainWindow state agree exactly
        during the live drag, not only after release.  The drag domain is the
        current marker domain ``[0, marker_upper]`` (analysis units,
        PHI-R3.1), so a white point above ``1`` can be dragged/set.
        """
        if not self._drag_line or not self.has_data:
            return
        upper = self._marker_upper
        level = min(max(self._x_to_level(x), 0.0), upper)
        if self._drag_line == "min":
            self._black_point = quantize_bp_wp(
                min(level, self._white_point - _MIN_SEPARATION),
                max_value=upper,
            )
        else:
            self._white_point = quantize_bp_wp(
                max(level, self._black_point + _MIN_SEPARATION),
                max_value=upper,
            )
        self.update()
        self._schedule_live_drag()

    def _schedule_live_drag(self) -> None:
        """Arm the coalescing timer (no-op if already armed)."""
        if not self._live_drag_timer.isActive():
            self._live_drag_timer.start()

    def _emit_live_drag(self) -> None:
        """Emit one coalesced live BP/WP update (timer slot / test seam)."""
        self._live_drag_timer.stop()
        if self._drag_line is not None and self.has_data:
            self.rangeChanged.emit(self._black_point, self._white_point)

    def _end_drag(self) -> None:
        """Finish a drag, emitting the exact final BP/WP on release."""
        self._live_drag_timer.stop()
        if self._drag_line and self.has_data:
            self._drag_line = None
            self.rangeChanged.emit(self._black_point, self._white_point)
        self._drag_line = None

    # ------------------------------------------------------------- events
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_drag_at(event.position().x())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_line is not None:
            self._drag_at(event.position().x())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._end_drag()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.expandRequested.emit()
        super().mouseDoubleClickEvent(event)

    # ------------------------------------------------------------- drawing
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKGROUND)
        if not self.has_data:
            painter.setPen(_EMPTY_TEXT_COLOR)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No preview")
            painter.end()
            return

        rect = self._plot_rect()
        if rect.width() <= 1.0 or rect.height() <= 1.0:
            painter.end()
            return

        heights, lo, hi, draw_overflow = self._bars_for_current_mode()
        self._paint_bars(painter, rect, heights, lo=lo, hi=hi)
        if draw_overflow:
            self._draw_overflow_marker(painter, rect)

        self._draw_line(painter, rect, self._black_point, _BLACK_POINT_COLOR)
        self._draw_line(painter, rect, self._white_point, _WHITE_POINT_COLOR)
        self._paint_axis_labels(painter, rect)
        painter.end()

    def _bars_for_current_mode(self):
        """Select the bars to draw (and their X domain) for the current mode.

        FRP-H1: in FULL view mode the surface paints the model's **full-domain
        histogram** — ``full_log_counts`` (fallback: ``log_counts`` for
        synthetic/legacy-compatible models without the key) spread over
        ``full_hist_range`` (fallback: the model analysis range ``_model_range``)
        — so Reset/Full shows the complete sampled distribution, tail included,
        and the overflow marker is NOT drawn (nothing is un-binned: the bars
        reach the true analysis maximum).  In every other mode the current
        behaviour is preserved: ``log_counts`` over the model ``bin_range``
        with the plot-top overflow marker when values exist above it.  Returns
        ``(heights, lo, hi, draw_overflow)``; the legacy path returns the
        historical 256-bin histogram over ``[0, 1]`` (no marker).
        """
        if self._model is not None:
            if self._view_mode == "full":
                heights = self._model.get("full_log_counts")
                if heights is None:
                    heights = self._model["log_counts"]
                lo, hi = self._full_bin_domain()
                return heights, lo, hi, False
            return (
                self._model["log_counts"],
                self._bin_range[0],
                self._bin_range[1],
                True,
            )
        # Legacy 256-bin linear counts over the display-level [0, 1].
        return self._histogram, 0.0, 1.0, False

    def _full_bin_domain(self) -> tuple:
        """X domain the full-domain bars live in (FRP-H1).

        The model's ``full_hist_range`` (== the full analysis range
        ``(0.0, upper)`` — the true maximum of the sampled distribution);
        synthetic models without the key fall back to the validated model
        analysis range.  Legacy data (no model) keeps ``[0, 1]``.
        """
        if self._model is not None:
            full_hist_range = self._model.get("full_hist_range")
            if full_hist_range is not None:
                return _validated_model_range(full_hist_range)
            return self._model_range
        return (0.0, 1.0)

    def _draw_overflow_marker(self, painter: QPainter, rect: QRectF) -> None:
        """Draw a truthful "plot top / overflow" marker at the bin-range high.

        PHI-AUTO-HISTOGRAM-UX-V1 UI indicator: when the model reports values
        above the plotted bin high (``overflow_total > 0``), a thin vertical
        marker is drawn at the plot-top level (``bin_range`` high) whenever
        that level is inside the current view window, so the user sees exactly
        where the binned curve ends; the count of values above it is available
        as :attr:`overflow_total` and in the model metadata (their extent also
        stays visible through the stats ``max`` / ``full_range``).  FULL view
        mode never invokes this (the full-domain bars reach the true analysis
        maximum, so nothing is un-binned there — see
        :meth:`_bars_for_current_mode`).  A no-op for models without overflow
        or when the boundary is outside the view.
        """
        if self._overflow_total <= 0 or self._model is None:
            return
        boundary = self._bin_range[1]
        if boundary < self._view_min or boundary > self._view_max:
            return
        x = self._level_to_x(boundary)
        painter.save()
        try:
            pen = QPen(_OVERFLOW_MARKER_COLOR, 1, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawLine(
                QPointF(x, rect.top()),
                QPointF(x, rect.bottom()),
            )
        finally:
            painter.restore()

    def _paint_bars(
        self,
        painter: QPainter,
        rect: QRectF,
        heights_by_name: Dict[str, Any],
        lo: float = 0.0,
        hi: float = 1.0,
    ) -> None:
        """Draw per-channel bars, normalising over the *visible* bins.

        Channels are composited *additively* so single R/G/B regions remain
        distinguishable and overlapping distributions remain visibly composite
        (pair overlaps blend toward yellow/magenta/cyan, the triple overlap
        toward white) instead of the last channel masking the earlier ones.
        The painter composition state is saved before the bars and restored
        before the BP/WP markers and axis labels are drawn.

        ``lo``/``hi`` declare the X domain the bin indices live in: the legacy
        path passes the display-level ``(0, 1)`` window, the float-model path
        passes the model's preserved analysis range (``hi`` can exceed ``1.0``
        when the analysis buffer carries HDR headroom — PHI-R3).  Bars whose
        level lies outside the current view window are not drawn.
        """
        span = hi - lo
        if span <= 0.0:
            span = 1.0

        def _bin_level(n: int, i: int) -> float:
            return lo + (i + 0.5) / n * span

        max_h = 1.0
        for h in heights_by_name.values():
            n = len(h)
            for i in range(n):
                center = _bin_level(n, i)
                if self._view_min <= center <= self._view_max:
                    max_h = max(max_h, float(h[i]))

        painter.save()
        try:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Plus
            )
            for name, h in heights_by_name.items():
                color = _CHANNEL_COLORS.get(name, QColor(225, 225, 225))
                painter.setPen(color)
                n = len(h)
                for i in range(n):
                    value = float(h[i])
                    if value <= 0.0:
                        continue
                    center = _bin_level(n, i)
                    if center < self._view_min or center > self._view_max:
                        continue
                    bar = int(round(value / max_h * (rect.height() - 1.0)))
                    x = self._level_to_x(center)
                    painter.drawLine(
                        QPointF(x, rect.bottom()),
                        QPointF(x, rect.bottom() - bar),
                    )
        finally:
            painter.restore()

    def _paint_axis_labels(self, painter: QPainter, rect: QRectF) -> None:
        """Label the plotted X domain (view window) under the bars."""
        label_h = max(1.0, self.height() - rect.bottom() - 2.0)
        if label_h < 8.0:
            return
        painter.setPen(_AXIS_TEXT_COLOR)
        font = painter.font()
        if font.pointSize() > 7:
            font.setPointSize(font.pointSize() - 2)
        painter.setFont(font)
        y = rect.bottom() + 1.0
        painter.drawText(
            QRectF(rect.left(), y, 48.0, label_h),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{self._view_min:.2f}",
        )
        painter.drawText(
            QRectF(rect.right() - 48.0, y, 48.0, label_h),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{self._view_max:.2f}",
        )

    def _draw_line(self, painter: QPainter, rect: QRectF, level: float, color: QColor) -> None:
        x = self._level_to_x(level)
        pen = QPen(color, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
