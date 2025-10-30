# zemosaic_align_stack.py
"""
╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Auteur  : Tinystork, seigneur des couteaux à beurre (aka Tristan Nauleau)  
║ Partenaire : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║              (aka ChatGPT, Grand Maître du ciselage de code)         ║
║                                                                      ║
║ Licence : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description :                                                        ║
║   Ce programme a été forgé à la lueur des pixels et de la caféine,   ║
║   dans le but noble de transformer des nuages de photons en art      ║
║   astronomique. Si vous l’utilisez, pensez à dire “merci”,           ║
║   à lever les yeux vers le ciel, ou à citer Tinystork et J.A.R.V.I.S.║
║   (le karma des développeurs en dépend).                             ║
║                                                                      ║
║ Avertissement :                                                      ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le       ║
║   développement de ce code.                                          ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Author  : Tinystork, Lord of the Butter Knives (aka Tristan Nauleau) ║
║ Partner : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║           (aka ChatGPT, Grand Master of Code Chiseling)              ║
║                                                                      ║
║ License : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description:                                                         ║
║   This program was forged under the sacred light of pixels and       ║
║   caffeine, with the noble intent of turning clouds of photons into  ║
║   astronomical art. If you use it, please consider saying “thanks,”  ║
║   gazing at the stars, or crediting Tinystork and J.A.R.V.I.S. —     ║
║   developer karma depends on it.                                     ║
║                                                                      ║
║ Disclaimer:                                                          ║
║   No AIs or butter knives were harmed in the making of this code.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""


import numpy as np
import os
import importlib.util
import tempfile
from dataclasses import dataclass
from typing import Optional, Callable, Any

GPU_AVAILABLE = importlib.util.find_spec("cupy") is not None
import traceback
import gc
import logging  # Added for logger fallback
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# dépendance Photutils
PHOTOUTILS_AVAILABLE = False
DAOStarFinder, FITSFixedWarning, CircularAperture, aperture_photometry, SigmaClip, Background2D, MedianBackground, SourceCatalog = [None]*8 # type: ignore
try:
    from astropy.stats import SigmaClip, gaussian_sigma_to_fwhm # gaussian_sigma_to_fwhm est utile
    from astropy.table import Table
    from photutils.detection import DAOStarFinder
    from photutils.aperture import CircularAperture, aperture_photometry
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import detect_sources, SourceCatalog

# ... et _internal_logger
    # Ignorer les avertissements FITSFixedWarning de photutils si besoin
    import warnings
    from astropy.wcs import FITSFixedWarning
    warnings.filterwarnings('ignore', category=FITSFixedWarning)
    
    PHOTOUTILS_AVAILABLE = True
    # print("INFO (zemosaic_align_stack): Photutils importé.")
except ImportError:
    print("AVERT (zemosaic_align_stack): Photutils non disponible. FWHM weighting limité.")

# --- Dépendance Astroalign ---
ASTROALIGN_AVAILABLE = False
astroalign_module = None 
try:
    import astroalign as aa
    astroalign_module = aa
    ASTROALIGN_AVAILABLE = True
    # print("INFO (zemosaic_align_stack): Astroalign importé.") # Log au démarrage du worker principal
except ImportError:
    print("ERREUR CRITIQUE (zemosaic_align_stack): Astroalign non installé. Alignement impossible.")

# --- Dépendance Astropy (pour sigma_clipped_stats) ---
SIGMA_CLIP_AVAILABLE = False
sigma_clipped_stats_func = None
try:
    from astropy.stats import sigma_clipped_stats
    sigma_clipped_stats_func = sigma_clipped_stats
    SIGMA_CLIP_AVAILABLE = True
    # print("INFO (zemosaic_align_stack): Astropy.stats.sigma_clipped_stats importé.")
except ImportError:
    print("AVERT (zemosaic_align_stack): Astropy.stats non disponible. Kappa-sigma stacking limité.")

# --- Dépendance SciPy (pour Winsorize) ---
SCIPY_AVAILABLE = False
winsorize_func = None
try:
    from scipy.stats.mstats import winsorize
    winsorize_func = winsorize
    SCIPY_AVAILABLE = True
    # print("INFO (zemosaic_align_stack): Scipy.stats.mstats.winsorize importé.")
except ImportError:
    print("AVERT (zemosaic_align_stack): Scipy non disponible. Winsorized Sigma Clip non fonctionnel.")

def _iter_row_chunks(total_rows: int, frames: int, width: int, itemsize: int,
                     max_chunk_bytes: int = 256 * 1024 * 1024):
    """Yield slices that limit the memory footprint of (N, H, W) chunks."""

    if total_rows <= 0:
        return

    if max_chunk_bytes <= 0:
        yield slice(0, total_rows)
        return

    bytes_per_row = frames * width * itemsize
    if bytes_per_row <= 0:
        yield slice(0, total_rows)
        return

    rows_per_chunk = max(1, min(total_rows, max_chunk_bytes // bytes_per_row))
    if rows_per_chunk <= 0:
        rows_per_chunk = 1

    for start in range(0, total_rows, rows_per_chunk):
        end = min(total_rows, start + rows_per_chunk)
        yield slice(start, end)


def _winsorize_block_numpy(arr_block: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Winsorize a spatial block along axis 0 without modifying the input."""

    low, high = limits
    block = arr_block.astype(np.float32, copy=False)
    result = block.copy()
    if low > 0:
        lower = np.quantile(block, low, axis=0)
        lower = lower.astype(np.float32, copy=False)
        np.maximum(result, lower, out=result)
    if high > 0:
        upper = np.quantile(block, 1.0 - high, axis=0)
        upper = upper.astype(np.float32, copy=False)
        np.minimum(result, upper, out=result)
    return result

ZEMOSAIC_UTILS_AVAILABLE_FOR_RADIAL = False
make_radial_weight_map_func = None
try:
    from zemosaic_utils import make_radial_weight_map
    make_radial_weight_map_func = make_radial_weight_map
    ZEMOSAIC_UTILS_AVAILABLE_FOR_RADIAL = True
except ImportError as e_util_rad:
        print(f"AVERT (zemosaic_align_stack): Radial weighting: Erreur import make_radial_weight_map: {e_util_rad}")

_internal_logger = logging.getLogger("zemosaic.align_stack")
if not _internal_logger.handlers:
    _internal_logger.addHandler(logging.NullHandler())

_STACK_LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "DEBUG_DETAIL": logging.DEBUG,
    "INFO": logging.INFO,
    "INFO_DETAIL": logging.DEBUG,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "SUCCESS": logging.INFO,
    "ETA_LEVEL": logging.DEBUG,
    "CHRONO_LEVEL": logging.DEBUG,
}

_STACK_USER_FACING_LEVELS = {"INFO", "WARN", "WARNING", "ERROR", "SUCCESS"}


def equalize_rgb_medians_inplace(img: np.ndarray) -> tuple[float, float, float, float]:
    """In-place per-channel median equalization so that median(R)==median(G)==median(B)."""

    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return (1.0, 1.0, 1.0, float("nan"))

    med = np.nanmedian(img, axis=(0, 1)).astype(np.float32)
    finite = np.isfinite(med) & (med > 0)
    if not np.any(finite):
        return (1.0, 1.0, 1.0, float("nan"))

    target = float(np.nanmedian(med[finite]))
    gains = np.ones(3, dtype=np.float32)
    gains[finite] = target / med[finite]
    img *= gains[None, None, :]
    return (float(gains[0]), float(gains[1]), float(gains[2]), target)


def _coerce_config_bool(value: Any, default: bool = True) -> bool:
    """Best-effort coercion of configuration values to booleans."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _poststack_rgb_equalization(
    stacked: np.ndarray | None,
    zconfig=None,
    stack_metadata: dict | None = None,
) -> dict:
    """Apply optional RGB equalization and record metadata."""

    default_enabled = True
    try:
        if isinstance(zconfig, dict):
            enabled = _coerce_config_bool(zconfig.get("poststack_equalize_rgb", default_enabled), default_enabled)
        else:
            enabled = _coerce_config_bool(getattr(zconfig, "poststack_equalize_rgb", default_enabled), default_enabled)
    except Exception:
        enabled = default_enabled

    info = {
        "enabled": enabled,
        "applied": False,
        "gain_r": 1.0,
        "gain_g": 1.0,
        "gain_b": 1.0,
        "target_median": float("nan"),
    }

    if not enabled or stacked is None or not isinstance(stacked, np.ndarray):
        if stack_metadata is not None:
            stack_metadata["rgb_equalization"] = info
        return info

    if stacked.ndim != 3 or stacked.shape[2] != 3:
        if stack_metadata is not None:
            stack_metadata["rgb_equalization"] = info
        return info

    try:
        gain_r, gain_g, gain_b, target = equalize_rgb_medians_inplace(stacked)
        info.update(
            gain_r=float(gain_r),
            gain_g=float(gain_g),
            gain_b=float(gain_b),
            target_median=float(target),
        )
        if np.isfinite(target):
            info["applied"] = True
            _internal_logger.info(
                "[RGB-EQ] Applied per-substack RGB median equalization: "
                f"gains (R,G,B)=({gain_r:.6f},{gain_g:.6f},{gain_b:.6f}), target_median={target:.6g}"
            )
        else:
            _internal_logger.info(
                "[RGB-EQ] Skipped RGB equalization due to non-finite channel medians."
            )
    except Exception as exc:
        _internal_logger.warning(f"[RGB-EQ] Skipped RGB equalization due to error: {exc}")
    finally:
        if stack_metadata is not None:
            stack_metadata["rgb_equalization"] = info

    return info


def _normalize_stack_output(result: Any) -> tuple[np.ndarray, float]:
    """Normalize heterogeneous stack outputs into (array, rejected_pct)."""

    if isinstance(result, (list, tuple)) and len(result) >= 1:
        stacked = np.asarray(result[0], dtype=np.float32).astype(np.float32, copy=False)
        rejected = float(result[1]) if len(result) > 1 else 0.0
    else:
        stacked = np.asarray(result, dtype=np.float32).astype(np.float32, copy=False)
        rejected = 0.0
    return stacked, rejected


def _log_stack_message(
    message_key_or_raw: Any,
    level: str | None = "INFO",
    progress_callback: Optional[Callable] = None,
    **kwargs: Any,
) -> None:
    """Forward stack logging messages to the worker log and optional callback.

    Parameters
    ----------
    message_key_or_raw:
        Either a localization key or a raw message string.
    level:
        Logging severity. Mirrors :func:`zemosaic_worker._log_and_callback`.
    progress_callback:
        Callback compatible with :func:`zemosaic_worker._log_and_callback`.
    **kwargs:
        Extra context forwarded to the callback and used for formatting when
        logging non user-facing levels.
    """

    level_str = "INFO"
    if isinstance(level, str):
        level_str = level.upper()
    elif level is not None:
        try:
            level_str = str(level).upper()
        except Exception:
            level_str = "INFO"

    if level_str == "WARNING":
        # Normalise to WARN to align with worker expectations.
        level_str = "WARN"

    if level_str in _STACK_USER_FACING_LEVELS:
        message_for_internal_log = f"[STACK][KEY_OR_RAW: {message_key_or_raw}]"
        if kwargs:
            message_for_internal_log += f" (Args: {kwargs})"
    else:
        message_for_internal_log = str(message_key_or_raw)
        if kwargs:
            try:
                message_for_internal_log = message_for_internal_log.format(**kwargs)
            except Exception:
                # Keep the unformatted message but append kwargs for context.
                message_for_internal_log = f"{message_for_internal_log} | kwargs={kwargs}"

    _internal_logger.log(
        _STACK_LOG_LEVEL_MAP.get(level_str, logging.INFO),
        message_for_internal_log,
    )

    if progress_callback and callable(progress_callback):
        try:
            progress_callback(message_key_or_raw, None, level_str, **kwargs)
        except Exception as cb_err:
            _internal_logger.warning(
                "Stack progress callback failure for %r: %s",
                message_key_or_raw,
                cb_err,
                exc_info=True,
            )

CPU_WINSOR_MEMMAP_THRESHOLD_BYTES = int(
    os.environ.get("ZEMOSAIC_CPU_WINSOR_MEMMAP_THRESHOLD", 3 * 1024**3)
)


def _cleanup_memmap(arr: np.ndarray, path: Optional[str]) -> None:
    """Release resources for a temporary memmap-backed array."""

    if isinstance(arr, np.memmap):
        try:
            arr._mmap.close()  # type: ignore[attr-defined]
        except Exception:
            pass
    if path:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            _internal_logger.debug(
                "CPU winsorized fallback: unable to delete memmap file",
                exc_info=True,
            )


@dataclass
class WinsorStreamingState:
    """Accumulate streaming statistics for winsorized sigma clip."""

    sum_values: np.ndarray
    count_values: np.ndarray
    fallback_sum: np.ndarray
    fallback_count: np.ndarray
    weighted_sum: Optional[np.ndarray] = None
    weighted_weight: Optional[np.ndarray] = None
    total_pixels: int = 0
    kept_pixels: int = 0

    @classmethod
    def create(cls, spatial_shape: tuple[int, ...]) -> "WinsorStreamingState":
        sum_values = np.zeros(spatial_shape, dtype=np.float64)
        count_values = np.zeros(spatial_shape, dtype=np.int64)
        fallback_sum = np.zeros(spatial_shape, dtype=np.float64)
        fallback_count = np.zeros(spatial_shape, dtype=np.int64)
        return cls(sum_values, count_values, fallback_sum, fallback_count)

    def update(
        self,
        processed_chunk: np.ndarray,
        fallback_sum_chunk: np.ndarray,
        fallback_count_chunk: np.ndarray,
        weights_chunk: Optional[np.ndarray] = None,
    ) -> None:
        processed_chunk = np.asarray(processed_chunk, dtype=np.float32)
        chunk_mask = np.isfinite(processed_chunk)
        self.sum_values += np.nansum(processed_chunk, axis=0, dtype=np.float64)
        self.count_values += chunk_mask.sum(axis=0, dtype=np.int64)
        self.total_pixels += int(processed_chunk.size)
        self.kept_pixels += int(np.count_nonzero(chunk_mask))

        self.fallback_sum += np.asarray(fallback_sum_chunk, dtype=np.float64)
        self.fallback_count += np.asarray(fallback_count_chunk, dtype=np.int64)

        if weights_chunk is not None:
            weights_broadcast = _broadcast_weights_for_chunk(weights_chunk, processed_chunk.shape)
            weights_broadcast = np.where(chunk_mask, weights_broadcast, 0.0)
            if self.weighted_sum is None or self.weighted_weight is None:
                self.weighted_sum = np.zeros(processed_chunk.shape[1:], dtype=np.float64)
                self.weighted_weight = np.zeros(processed_chunk.shape[1:], dtype=np.float64)
            self.weighted_sum += np.nansum(processed_chunk * weights_broadcast, axis=0, dtype=np.float64)
            self.weighted_weight += np.sum(weights_broadcast, axis=0, dtype=np.float64)

    def _fallback_mean(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            fallback = np.divide(
                self.fallback_sum,
                self.fallback_count,
                out=np.zeros_like(self.fallback_sum, dtype=np.float64),
                where=self.fallback_count > 0,
            )
        return np.where(self.fallback_count > 0, fallback, np.nan)

    def finalize(self, use_weights: bool) -> tuple[np.ndarray, float]:
        fallback = self._fallback_mean()
        stacked: np.ndarray
        if use_weights and self.weighted_sum is not None and self.weighted_weight is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                stacked = np.divide(
                    self.weighted_sum,
                    self.weighted_weight,
                    out=np.zeros_like(fallback, dtype=np.float64),
                    where=self.weighted_weight > 0,
                )
            needs_fallback = ~np.isfinite(stacked)
            if np.any(needs_fallback):
                stacked = np.where(needs_fallback, fallback, stacked)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                stacked = np.divide(
                    self.sum_values,
                    self.count_values,
                    out=np.zeros_like(fallback, dtype=np.float64),
                    where=self.count_values > 0,
                )
            zero_mask = self.count_values <= 0
            if np.any(zero_mask):
                stacked = np.where(zero_mask, fallback, stacked)
            needs_fallback = ~np.isfinite(stacked)
            if np.any(needs_fallback):
                stacked = np.where(needs_fallback, fallback, stacked)

        rejected_pct = 0.0
        if self.total_pixels > 0:
            rejected_pct = 100.0 * (self.total_pixels - self.kept_pixels) / float(self.total_pixels)

        return stacked.astype(np.float32, copy=False), float(rejected_pct)


def _broadcast_weights_for_chunk(weights_chunk: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast per-frame weights to match a chunk of stack data."""

    weights_chunk = np.asarray(weights_chunk, dtype=np.float32)
    if weights_chunk.shape[0] != target_shape[0]:
        raise ValueError(
            f"weights shape {weights_chunk.shape} incompatible with chunk leading dimension {target_shape[0]}"
        )
    if weights_chunk.ndim == 1:
        extra_dims = (1,) * (len(target_shape) - 1)
        weights_chunk = weights_chunk.reshape((target_shape[0],) + extra_dims)
    try:
        broadcast = np.broadcast_to(weights_chunk, target_shape)
    except ValueError as exc:
        raise ValueError(
            f"weights shape {weights_chunk.shape} not compatible with frames shape {target_shape}"
        ) from exc
    return broadcast.astype(np.float64, copy=False)


# Optional access to utils for GPU helpers
ZU_AVAILABLE = False
zutils = None
try:
    import zemosaic_utils as zutils  # type: ignore
    ZU_AVAILABLE = True
except Exception:
    ZU_AVAILABLE = False

if ZU_AVAILABLE:
    _ensure_pool = getattr(zutils, "ensure_cupy_pool_initialized", None)
    _free_pools = getattr(zutils, "free_cupy_memory_pools", None)
    _gpu_memory_ok = getattr(zutils, "gpu_memory_sufficient", None)
else:
    _ensure_pool = None
    _free_pools = None
    _gpu_memory_ok = None


def _callable_or_none(func):
    return func if callable(func) else None


def _ensure_gpu_pool():
    func = _callable_or_none(_ensure_pool)
    if func is not None:
        try:
            func()
        except Exception:
            pass


def _free_gpu_pools():
    func = _callable_or_none(_free_pools)
    if func is not None:
        try:
            func()
        except Exception:
            pass


def _gpu_nanpercentile(values: np.ndarray, percentiles):
    """Compute nan-aware percentiles on the GPU and release cached memory."""

    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy is not available")

    arr_gpu = None
    try:
        import cupy as cp  # type: ignore

        _ensure_gpu_pool()
        arr_gpu = cp.asarray(values, dtype=cp.float32)
        result_gpu = cp.nanpercentile(arr_gpu, percentiles)

        if np.isscalar(percentiles):
            return float(result_gpu)

        result_np = cp.asnumpy(result_gpu)
        return np.asarray(result_np, dtype=np.float64)
    finally:
        if arr_gpu is not None:
            del arr_gpu
        _free_gpu_pools()


def _has_gpu_budget(estimated_bytes: int) -> bool:
    func = _callable_or_none(_gpu_memory_ok)
    if func is None:
        return True
    try:
        return bool(func(int(estimated_bytes), safety_fraction=0.75))
    except Exception:
        return True

# --- Import des méthodes de stack CPU provenant du projet Seestar ---
cpu_stack_winsorized = None
cpu_stack_kappa = None
cpu_stack_linear = None
try:
    import importlib.util, os, pathlib
    _sm_path = pathlib.Path(__file__).resolve().parents[1] / 'seestar' / 'core' / 'stack_methods.py'
    if _sm_path.exists():
        spec = importlib.util.spec_from_file_location('seestar_stack_methods', _sm_path)
        _sm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_sm)  # type: ignore
        cpu_stack_winsorized = getattr(_sm, '_stack_winsorized_sigma', None)
        cpu_stack_kappa = getattr(_sm, '_stack_kappa_sigma', None)
        cpu_stack_linear = getattr(_sm, '_stack_linear_fit_clip', None)
except Exception as e_import_stack:
    print(f"AVERT (zemosaic_align_stack): Optional import of external stack_methods failed: {e_import_stack}")

# --- Implementations GPU simplifiées des méthodes de stack ---
def gpu_stack_winsorized(frames, *, kappa=3.0, winsor_limits=(0.05, 0.05), apply_rewinsor=True):
    """GPU Winsorized Sigma-Clip aligned with CPU logic.

    Order matches CPU: winsor -> sigma (median/std on winsorized) -> mean.
    Rejection mask is computed on original data using bounds from winsorized stats.
    """
    import cupy as cp

    frames_np = [np.asarray(f, dtype=np.float32) for f in frames if f is not None]
    if not frames_np:
        raise ValueError("No frames provided")

    per_frame_bytes = frames_np[0].nbytes
    estimated_bytes = per_frame_bytes * len(frames_np) * 4
    if not _has_gpu_budget(estimated_bytes):
        raise RuntimeError("GPU winsorized clip: insufficient memory budget")

    _ensure_gpu_pool()

    try:
        arr = cp.stack([cp.asarray(f, dtype=cp.float32) for f in frames_np], axis=0)

        low, high = float(winsor_limits[0]), float(winsor_limits[1])
        # Winsorize along stack axis (0) with NaN-safe quantiles
        q_low = cp.nanquantile(arr, low, axis=0)
        q_high = cp.nanquantile(arr, 1.0 - high, axis=0)
        arr_w = cp.clip(arr, q_low, q_high)

        # Sigma stats from winsorized data (median/std)
        median_w = cp.nanmedian(arr_w, axis=0)
        std_w = cp.nanstd(arr_w, axis=0)
        lower = median_w - (kappa * std_w)
        upper = median_w + (kappa * std_w)

        # Reject original values outside bounds; optionally re-winsorize rejected
        mask = (arr >= lower) & (arr <= upper)
        if apply_rewinsor:
            arr_clip = cp.where(mask, arr, arr_w)
        else:
            arr_clip = cp.where(mask, arr, cp.nan)

        result = cp.nanmean(arr_clip, axis=0)
        # Optional safety clip to [0,1] can be applied by caller if needed
        # result = cp.clip(result, 0.0, 1.0)

        rejected = 100.0 * float(mask.size - cp.count_nonzero(mask)) / float(mask.size)
        return cp.asnumpy(result.astype(cp.float32)), float(rejected)
    finally:
        _free_gpu_pools()


def gpu_stack_kappa(frames, *, sigma_low=3.0, sigma_high=3.0):
    import cupy as cp

    frames_np = [np.asarray(f, dtype=np.float32) for f in frames]
    if not frames_np:
        raise ValueError("No frames provided")

    per_frame_bytes = frames_np[0].nbytes
    estimated_bytes = per_frame_bytes * len(frames_np) * 3
    if not _has_gpu_budget(estimated_bytes):
        raise RuntimeError("GPU kappa clip: insufficient memory budget")

    _ensure_gpu_pool()

    try:
        arr = cp.stack([cp.asarray(f) for f in frames_np], axis=0)
        med = cp.nanmedian(arr, axis=0)
        std = cp.nanstd(arr, axis=0)
        low = med - sigma_low * std
        high = med + sigma_high * std
        mask = (arr >= low) & (arr <= high)
        arr_clip = cp.where(mask, arr, cp.nan)
        result = cp.nanmean(arr_clip, axis=0)
        rejected = 100.0 * float(mask.size - cp.count_nonzero(mask)) / float(mask.size)
        return cp.asnumpy(result.astype(cp.float32)), float(rejected)
    finally:
        _free_gpu_pools()


def gpu_stack_linear(frames, *, sigma=3.0):
    import cupy as cp

    frames_np = [np.asarray(f, dtype=np.float32) for f in frames]
    if not frames_np:
        raise ValueError("No frames provided")

    per_frame_bytes = frames_np[0].nbytes
    estimated_bytes = per_frame_bytes * len(frames_np) * 4
    if not _has_gpu_budget(estimated_bytes):
        raise RuntimeError("GPU linear clip: insufficient memory budget")

    _ensure_gpu_pool()

    try:
        arr = cp.stack([cp.asarray(f) for f in frames_np], axis=0)
        med = cp.nanmedian(arr, axis=0)
        resid = arr - med
        med_r = cp.nanmedian(resid, axis=0)
        std_r = cp.nanstd(resid, axis=0)
        mask = cp.abs(resid - med_r) <= sigma * std_r
        arr_clip = cp.where(mask, arr, cp.nan)
        result = cp.nanmean(arr_clip, axis=0)
        rejected = 100.0 * float(mask.size - cp.count_nonzero(mask)) / float(mask.size)
        return cp.asnumpy(result.astype(cp.float32)), float(rejected)
    finally:
        _free_gpu_pools()


def stack_winsorized_sigma_clip(
    frames,
    weights=None,
    weight_method: str = "none",
    zconfig=None,
    stack_metadata: dict | None = None,
    **kwargs,
):
    """
    Wrapper calling GPU or CPU winsorized sigma clip, with robust GPU guards.

    - La voie GPU ignore les `weights` (non supportés).
    - Si la voie GPU échoue ou produit une sortie suspecte, fallback CPU.
    - La voie CPU accepte `weights` en mot-clé si fournis.
    """
    # --- validations légères d'entrée ---
    if frames is None:
        raise ValueError("frames is None")
    import numpy as _np
    frames_list = list(frames)
    if not frames_list:
        raise ValueError("frames is empty")
    sample = _np.asarray(frames_list[0])
    if sample.ndim not in (2, 3):
        raise ValueError(f"each frame must be (H,W) or (H,W,C); got shape {sample.shape}")
    # Harmonize shapes for GPU path: drop mismatched frames early
    frames_list = [f for f in frames_list if _np.asarray(f).shape == sample.shape]
    if len(frames_list) < 3:
        _internal_logger.warning("Winsorized clip needs >=3 images; forcing CPU.")
        use_gpu = False
    else:
        # Accept either a generic 'use_gpu' flag or legacy 'use_gpu_phase5'
        if zconfig:
            # Stacking should not be toggled by the Phase‑5 GPU option.
            # Honor only explicit stacking flags or generic 'use_gpu'.
            use_gpu = bool(getattr(zconfig, 'stack_use_gpu',
                                   getattr(zconfig, 'use_gpu_stack',
                                           getattr(zconfig, 'use_gpu', False))))
        else:
            use_gpu = False

    progress_callback = kwargs.get("progress_callback")
    requested_weight_method = str(weight_method or "none").lower().strip()
    weight_method_in_use = requested_weight_method or "none"
    manual_weights = weights is not None
    weight_label_for_log = "custom" if manual_weights and weight_method_in_use == "none" else weight_method_in_use
    weights_array_full: _np.ndarray | None = None
    weight_stats: dict[str, float] | None = None

    if manual_weights:
        weights_array_full = _np.asarray(weights, dtype=_np.float32)
        if weights_array_full.shape[0] != len(frames_list):
            raise ValueError(
                f"weights shape {weights_array_full.shape} incompatible with frame count {len(frames_list)}"
            )
    elif weight_method_in_use not in ("", "none"):
        frames_np_for_weights = [_np.asarray(f, dtype=_np.float32) for f in frames_list]
        derived_list, effective_method, quality_stats = _compute_quality_weights(
            frames_np_for_weights,
            weight_method_in_use,
            progress_callback=progress_callback,
        )
        weight_method_in_use = effective_method
        if derived_list:
            sanitized_arrays = []
            for idx, arr in enumerate(derived_list):
                frame_shape = _np.asarray(frames_np_for_weights[idx]).shape if idx < len(frames_np_for_weights) else None
                if arr is not None:
                    w_arr = _np.asarray(arr, dtype=_np.float32, copy=False)
                    # Ensure compact per-frame weight: (1,1,C) for color or (1,) for mono/scalar
                    if frame_shape is not None and len(frame_shape) == 3 and frame_shape[-1] == 3:
                        if w_arr.ndim == 0:
                            w_arr = _np.full((1, 1, 3), float(w_arr), dtype=_np.float32)
                        elif w_arr.ndim == 1 and w_arr.shape[0] == 3:
                            w_arr = w_arr.reshape((1, 1, 3))
                        else:
                            # Last resort: reduce to per-channel mean and broadcast to (1,1,3)
                            try:
                                chv = _np.nanmean(w_arr, axis=(0, 1)) if w_arr.ndim >= 2 else _np.asarray(w_arr)
                                chv = _np.asarray(chv, dtype=_np.float32).reshape((1, 1, -1))
                                if chv.shape[-1] == 3:
                                    w_arr = chv
                                else:
                                    w_arr = _np.ones((1, 1, 3), dtype=_np.float32)
                            except Exception:
                                w_arr = _np.ones((1, 1, 3), dtype=_np.float32)
                    else:
                        # Monochrome target: scalar as (1,)
                        if w_arr.ndim > 0:
                            try:
                                val = float(_np.nanmean(w_arr))
                            except Exception:
                                val = 1.0
                            w_arr = _np.asarray([val], dtype=_np.float32)
                        else:
                            w_arr = w_arr.reshape((1,))
                    sanitized_arrays.append(w_arr)
                else:
                    # Default unit weight in compact form
                    if frame_shape is not None and len(frame_shape) == 3 and frame_shape[-1] == 3:
                        sanitized_arrays.append(_np.ones((1, 1, 3), dtype=_np.float32))
                    else:
                        sanitized_arrays.append(_np.ones((1,), dtype=_np.float32))
            # Stack into (N, 1[,1[,C]])
            weights_array_full = _np.stack(sanitized_arrays, axis=0)
            weight_stats = quality_stats
        del frames_np_for_weights

    if weights_array_full is not None:
        # Keep compact shape and defer broadcasting to the compute sites
        weights_array_full = _np.asarray(weights_array_full, dtype=_np.float32, copy=False)
        if weights_array_full.ndim == 1:
            # (N,) -> (N,1)
            weights_array_full = weights_array_full.reshape((weights_array_full.shape[0], 1))
        if weight_stats is None:
            try:
                weight_stats = {
                    "min": float(_np.nanmin(weights_array_full)),
                    "max": float(_np.nanmax(weights_array_full)),
                    "frames": float(len(frames_list)),
                }
            except Exception:
                weight_stats = None

    max_frames_per_pass = kwargs.pop("winsor_max_frames_per_pass", None)
    if max_frames_per_pass is None and zconfig is not None:
        max_frames_per_pass = getattr(zconfig, "winsor_max_frames_per_pass", 0)
    try:
        max_frames_per_pass = int(max_frames_per_pass)
    except (TypeError, ValueError):
        max_frames_per_pass = 0
    if max_frames_per_pass < 0:
        max_frames_per_pass = 0

    def _stack_winsor_streaming(limit: int) -> tuple[_np.ndarray, float]:
        nonlocal frames_list, weights_array_full

        winsor_limits = kwargs.get("winsor_limits", (0.05, 0.05))
        kappa = kwargs.get("kappa", 3.0)
        apply_rewinsor = kwargs.get("apply_rewinsor", True)
        winsor_max_workers = kwargs.get("winsor_max_workers", 1)
        progress_callback = kwargs.get("progress_callback")

        state = WinsorStreamingState.create(sample.shape)

        total_frames = len(frames_list)
        for start in range(0, total_frames, limit):
            stop = min(total_frames, start + limit)
            chunk = [_np.asarray(f, dtype=_np.float32) for f in frames_list[start:stop]]
            if not chunk:
                continue
            chunk_arr = _np.stack(chunk, axis=0)
            weights_chunk = None
            if weights_array_full is not None:
                weights_chunk = weights_array_full[start:stop]

            state, _ = _reject_outliers_winsorized_sigma_clip(
                stacked_array_NHDWC=chunk_arr,
                winsor_limits_tuple=winsor_limits,
                sigma_low=float(kappa),
                sigma_high=float(kappa),
                progress_callback=progress_callback,
                max_workers=int(winsor_max_workers or 1),
                apply_rewinsor=bool(apply_rewinsor),
                weights_chunk=weights_chunk,
                streaming_state=state,
            )

            # Libérer les références intermédiaires au plus tôt
            del chunk_arr, chunk

        stacked_stream, rejected_pct = state.finalize(weights_array_full is not None)
        stacked_stream = stacked_stream.astype(_np.float32, copy=False)
        _poststack_rgb_equalization(stacked_stream, zconfig, stack_metadata)
        return stacked_stream, rejected_pct

    if max_frames_per_pass and len(frames_list) > max_frames_per_pass:
        if use_gpu:
            _internal_logger.info(
                "Winsorized streaming enabled for %d frames (limit %d); forcing CPU path.",
                len(frames_list),
                max_frames_per_pass,
            )
            use_gpu = False
        else:
            _internal_logger.info(
                "Winsorized streaming enabled for %d frames (limit %d).",
                len(frames_list),
                max_frames_per_pass,
            )
        return _stack_winsor_streaming(max_frames_per_pass)

    # --- GPU path (poids ignorés) ---
    if use_gpu and GPU_AVAILABLE and callable(globals().get("gpu_stack_winsorized", None)):
        try:
            if weights_array_full is not None or weight_method_in_use not in ("", "none") or manual_weights:
                _log_stack_message(
                    f"[Stack][GPU Winsorized] weight_method='{weight_label_for_log}' requested but not supported on this path -> ignoring weights.",
                    "INFO",
                    progress_callback,
                )
            gpu_out = gpu_stack_winsorized(frames_list, **kwargs)

            # --- validations de sortie GPU ---
            if gpu_out is None:
                raise RuntimeError("GPU returned None")
            # Support both (image, rejected_pct) and image-only returns
            if isinstance(gpu_out, (list, tuple)) and len(gpu_out) >= 1:
                _gpu_img = gpu_out[0]
                _gpu_rej = float(gpu_out[1]) if len(gpu_out) > 1 else 0.0
            else:
                _gpu_img = gpu_out
                _gpu_rej = 0.0
            gpu_out = _np.asarray(_gpu_img, dtype=_np.float32)
            # Sortie attendue: même champs spatiaux que frames_np sans l’axe N
            exp_shape = sample.shape  # (H,W) ou (H,W,C)
            if gpu_out.shape != exp_shape:
                raise RuntimeError(f"GPU returned shape {gpu_out.shape}, expected {exp_shape}")
            if not _np.any(_np.isfinite(gpu_out)):
                raise RuntimeError("GPU output has no finite values")
            # tolérance: > 90% de pixels finis
            finite_ratio = _np.isfinite(gpu_out).mean()
            if finite_ratio < 0.9:
                raise RuntimeError(f"GPU output has too many NaN/Inf (finite_ratio={finite_ratio:.2%})")

            _poststack_rgb_equalization(gpu_out, zconfig, stack_metadata)
            return gpu_out, float(_gpu_rej)

        except Exception as e:
            _internal_logger.warning(
                f"GPU winsorized clip failed or looked invalid → fallback CPU: {type(e).__name__}: {e}",
                exc_info=True
            )

    # --- CPU path ---
    if not callable(globals().get("cpu_stack_winsorized", None)):
        raise RuntimeError("CPU stack_winsorized function unavailable")

    cpu_kwargs = dict(kwargs)
    if weights_array_full is not None:
        cpu_kwargs["weights"] = weights_array_full

    try:
        result = cpu_stack_winsorized(frames_list, **cpu_kwargs)
    except TypeError:
        if weights_array_full is not None:
            _log_stack_message(
                "[Stack][Winsorized CPU] External implementation rejected weights; falling back to internal handler.",
                "WARN",
                progress_callback,
            )
            result = _cpu_stack_winsorized_fallback(frames_list, **cpu_kwargs)
        else:
            raise

    if weights_array_full is not None and weight_stats:
        _log_stack_message(
            f"[Stack][Winsorized CPU] weight_method='{weight_label_for_log}'; weights: min={weight_stats['min']:.3g} max={weight_stats['max']:.3g}",
            "INFO",
            progress_callback,
        )

    stacked_cpu, rejected = _normalize_stack_output(result)
    _poststack_rgb_equalization(stacked_cpu, zconfig, stack_metadata)
    return stacked_cpu, rejected


def stack_kappa_sigma_clip(
    frames,
    weight_method: str = "none",
    zconfig=None,
    stack_metadata: dict | None = None,
    **kwargs,
):
    """Wrapper calling GPU or CPU kappa-sigma clip.

    Honors a generic ``use_gpu`` flag on ``zconfig`` if present, otherwise
    falls back to the legacy ``use_gpu_phase5`` flag used by the GUI.
    """
    # Do not let the Phase‑5 GPU flag affect stacking.
    use_gpu = (getattr(zconfig, 'stack_use_gpu',
                       getattr(zconfig, 'use_gpu_stack',
                               getattr(zconfig, 'use_gpu', False)))
               if zconfig else False)
    progress_callback = kwargs.get("progress_callback")
    frames_list = list(frames)
    requested_weight_method = str(weight_method or "none").lower().strip()
    weight_method_in_use = requested_weight_method or "none"
    weights_array: np.ndarray | None = None
    weight_stats: dict[str, float] | None = None

    if requested_weight_method not in ("", "none"):
        frames_np_for_weights = [np.asarray(f, dtype=np.float32) for f in frames_list]
        derived_list, effective_method, quality_stats = _compute_quality_weights(
            frames_np_for_weights,
            requested_weight_method,
            progress_callback=progress_callback,
        )
        weight_method_in_use = effective_method
        if derived_list:
            sanitized_arrays = []
            for idx, arr in enumerate(derived_list):
                if arr is not None:
                    sanitized_arrays.append(np.asarray(arr, dtype=np.float32))
                else:
                    sanitized_arrays.append(np.ones_like(frames_np_for_weights[idx], dtype=np.float32))
            weights_array = np.stack(sanitized_arrays, axis=0)
            weight_stats = quality_stats
        del frames_np_for_weights

    if use_gpu and GPU_AVAILABLE:
        try:
            if weights_array is not None:
                _log_stack_message(
                    f"[Stack][GPU Kappa] weight_method='{weight_method_in_use}' requested but not supported on GPU path; ignoring weights.",
                    "INFO",
                    progress_callback,
                )
            gpu_result = gpu_stack_kappa(frames_list, **kwargs)
            stacked_gpu, rejected_gpu = _normalize_stack_output(gpu_result)
            _poststack_rgb_equalization(stacked_gpu, zconfig, stack_metadata)
            return stacked_gpu, rejected_gpu
        except Exception:
            _internal_logger.warning("GPU kappa clip failed, fallback CPU", exc_info=True)
    if cpu_stack_kappa:
        cpu_kwargs = dict(kwargs)
        if weights_array is not None:
            cpu_kwargs["weights"] = weights_array
        try:
            result = cpu_stack_kappa(frames_list, **cpu_kwargs)
        except TypeError:
            if weights_array is not None:
                _log_stack_message(
                    "[Stack][CPU Kappa] External implementation rejected weights; using fallback.",
                    "WARN",
                    progress_callback,
                )
                result = _cpu_stack_kappa_fallback(frames_list, **cpu_kwargs)
            else:
                raise
        else:
            if weights_array is not None and weight_stats:
                _log_stack_message(
                    f"[Stack] Using CPU kappa-sigma; weight_method='{weight_method_in_use}'; weights: min={weight_stats['min']:.3g} max={weight_stats['max']:.3g}",
                    "INFO",
                    progress_callback,
                )
        stacked_cpu, rejected_cpu = _normalize_stack_output(result)
        _poststack_rgb_equalization(stacked_cpu, zconfig, stack_metadata)
        return stacked_cpu, rejected_cpu
    fallback_result = _cpu_stack_kappa_fallback(frames_list, **kwargs)
    stacked_fb, rejected_fb = _normalize_stack_output(fallback_result)
    _poststack_rgb_equalization(stacked_fb, zconfig, stack_metadata)
    return stacked_fb, rejected_fb


def stack_linear_fit_clip(
    frames,
    weight_method: str = "none",
    zconfig=None,
    stack_metadata: dict | None = None,
    **kwargs,
):
    """Wrapper calling GPU or CPU linear fit clip.

    Honors a generic ``use_gpu`` flag on ``zconfig`` if present, otherwise
    falls back to the legacy ``use_gpu_phase5`` flag used by the GUI.
    """
    # Do not let the Phase‑5 GPU flag affect stacking.
    use_gpu = (getattr(zconfig, 'stack_use_gpu',
                       getattr(zconfig, 'use_gpu_stack',
                               getattr(zconfig, 'use_gpu', False)))
               if zconfig else False)
    progress_callback = kwargs.get("progress_callback")
    frames_list = list(frames)
    requested_weight_method = str(weight_method or "none").lower().strip()
    weight_method_in_use = requested_weight_method or "none"
    weights_array: np.ndarray | None = None
    weight_stats: dict[str, float] | None = None

    if requested_weight_method not in ("", "none"):
        frames_np_for_weights = [np.asarray(f, dtype=np.float32) for f in frames_list]
        derived_list, effective_method, quality_stats = _compute_quality_weights(
            frames_np_for_weights,
            requested_weight_method,
            progress_callback=progress_callback,
        )
        weight_method_in_use = effective_method
        if derived_list:
            sanitized_arrays = []
            for idx, arr in enumerate(derived_list):
                if arr is not None:
                    sanitized_arrays.append(np.asarray(arr, dtype=np.float32))
                else:
                    sanitized_arrays.append(np.ones_like(frames_np_for_weights[idx], dtype=np.float32))
            weights_array = np.stack(sanitized_arrays, axis=0)
            weight_stats = quality_stats
        del frames_np_for_weights

    if use_gpu and GPU_AVAILABLE:
        try:
            if weights_array is not None:
                _log_stack_message(
                    f"[Stack][GPU Linear] weight_method='{weight_method_in_use}' requested but not supported on GPU path; ignoring weights.",
                    "INFO",
                    progress_callback,
                )
            gpu_result = gpu_stack_linear(frames_list, **kwargs)
            stacked_gpu, rejected_gpu = _normalize_stack_output(gpu_result)
            _poststack_rgb_equalization(stacked_gpu, zconfig, stack_metadata)
            return stacked_gpu, rejected_gpu
        except Exception:
            _internal_logger.warning("GPU linear clip failed, fallback CPU", exc_info=True)
    if cpu_stack_linear:
        cpu_kwargs = dict(kwargs)
        if weights_array is not None:
            cpu_kwargs["weights"] = weights_array
        try:
            result = cpu_stack_linear(frames_list, **cpu_kwargs)
        except TypeError:
            if weights_array is not None:
                _log_stack_message(
                    "[Stack][CPU Linear] External implementation rejected weights; using fallback.",
                    "WARN",
                    progress_callback,
                )
                result = _cpu_stack_linear_fallback(frames_list, **cpu_kwargs)
            else:
                raise
        else:
            if weights_array is not None and weight_stats:
                _log_stack_message(
                    f"[Stack] Using CPU linear-fit clip; weight_method='{weight_method_in_use}'; weights: min={weight_stats['min']:.3g} max={weight_stats['max']:.3g}",
                    "INFO",
                    progress_callback,
                )
        stacked_cpu, rejected_cpu = _normalize_stack_output(result)
        _poststack_rgb_equalization(stacked_cpu, zconfig, stack_metadata)
        return stacked_cpu, rejected_cpu
    fallback_result = _cpu_stack_linear_fallback(frames_list, **kwargs)
    stacked_fb, rejected_fb = _normalize_stack_output(fallback_result)
    _poststack_rgb_equalization(stacked_fb, zconfig, stack_metadata)
    return stacked_fb, rejected_fb


def _calculate_robust_stats_for_linear_fit(image_data_2d_float32: np.ndarray,
                                           low_percentile: float = 25.0,
                                           high_percentile: float = 90.0,
                                           progress_callback: callable = None,
                                           use_gpu: bool = False):
    """
    Calcule des statistiques robustes (deux points de percentiles) pour une image 2D (un canal).
    Utilisé par la normalisation par ajustement linéaire pour estimer le fond de ciel et
    un point légèrement au-dessus, tout en essayant d'éviter les étoiles brillantes.

    Args:
        image_data_2d_float32 (np.ndarray): Image 2D (un canal), dtype float32.
        low_percentile (float): Percentile inférieur (ex: 25.0 pour le fond de ciel).
        high_percentile (float): Percentile supérieur (ex: 90.0 pour un point au-dessus du fond).
        progress_callback (callable, optional): Fonction de callback pour les logs.

    Returns:
        tuple[float, float]: (stat_low, stat_high). Retourne (0.0, 1.0) en cas d'erreur majeure.
    """
    # Define a local alias for the callback for brevity and safety
    # Uses _internal_logger as a fallback if progress_callback is None
    _pcb = lambda msg_key, lvl="DEBUG_VERY_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    if not isinstance(image_data_2d_float32, np.ndarray) or image_data_2d_float32.ndim != 2:
        _pcb("stathelper_error_invalid_input_for_stats", lvl="WARN",
             shape=image_data_2d_float32.shape if hasattr(image_data_2d_float32, 'shape') else 'N/A',
             ndim=image_data_2d_float32.ndim if hasattr(image_data_2d_float32, 'ndim') else 'N/A')
        return 0.0, 1.0 # Fallback pour une entrée clairement incorrecte

    if image_data_2d_float32.size == 0:
        _pcb("stathelper_error_empty_image_for_stats", lvl="WARN")
        return 0.0, 1.0

    # Assurer que les données sont finies pour le calcul des percentiles
    # np.nanpercentile gère déjà les NaNs, mais il est bon de savoir si tout est non-fini.
    finite_data = image_data_2d_float32[np.isfinite(image_data_2d_float32)]
    if finite_data.size == 0:
        _pcb("stathelper_warn_all_nan_or_inf_for_stats", lvl="WARN")
        return 0.0, 1.0 # Pas de données valides pour calculer les percentiles

    try:
        # Prefer GPU percentiles when requested and available
        if use_gpu and GPU_AVAILABLE:
            try:
                if ZU_AVAILABLE and hasattr(zutils, "_percentiles_gpu"):
                    stat_low, stat_high = zutils._percentiles_gpu(
                        image_data_2d_float32, low_percentile, high_percentile
                    )  # type: ignore[attr-defined]
                else:
                    stats = _gpu_nanpercentile(
                        image_data_2d_float32,
                        [low_percentile, high_percentile],
                    )
                    stat_low, stat_high = float(stats[0]), float(stats[1])
            except Exception:
                stat_low = float(np.nanpercentile(image_data_2d_float32, low_percentile))
                stat_high = float(np.nanpercentile(image_data_2d_float32, high_percentile))
        else:
            stat_low = float(np.nanpercentile(image_data_2d_float32, low_percentile))
            stat_high = float(np.nanpercentile(image_data_2d_float32, high_percentile))
        # _pcb(f"stathelper_debug_percentiles_calculated: low_p={low_percentile}% -> {stat_low:.3g}, high_p={high_percentile}% -> {stat_high:.3g}", lvl="DEBUG_VERY_DETAIL")

    except Exception as e_perc:
        _pcb(f"stathelper_error_percentile_calc: {e_perc}", lvl="WARN")
        # Fallback très simple si nanpercentile échoue pour une raison imprévue
        # (normalement, ne devrait pas arriver si finite_data n'est pas vide)
        # Utilise les min/max des données finies comme un pis-aller.
        if finite_data.size > 0 : # Double check, bien que déjà fait avant
             stat_low = float(np.min(finite_data))
             stat_high = float(np.max(finite_data))
             _pcb(f"stathelper_warn_percentile_exception_fallback_minmax: low={stat_low:.3g}, high={stat_high:.3g}", lvl="WARN")
        else: # Ne devrait jamais être atteint si la logique précédente est correcte
            return 0.0, 1.0


    # Gérer le cas où l'image est (presque) plate
    if abs(stat_high - stat_low) < 1e-5: # 1e-5 est un seuil arbitraire, pourrait être ajusté
        _pcb(f"stathelper_warn_stats_nearly_equal: low={stat_low:.3g}, high={stat_high:.3g}. L'image est peut-être plate ou a peu de dynamique dans les percentiles choisis.", lvl="DEBUG_DETAIL")
        # Si les stats sont égales, cela signifie que la plage entre low_percentile et high_percentile est très étroite.
        # Cela peut arriver avec des images avec peu de signal ou très bruitées où les percentiles tombent au même endroit.
        # On retourne quand même ces valeurs, la logique appelante devra gérer cela (ex: a = 1, b = offset).

    return stat_low, stat_high



def align_images_in_group(image_data_list: list,
                          reference_image_index: int = 0,
                          detection_sigma: float = 3.0,
                          min_area: int = 5,
                          propagate_mask: bool = False,
                          progress_callback: callable = None) -> tuple[list, list[int]]:
    """
    Aligne une liste d'images (données NumPy HWC, float32, ADU) sur une image de référence
    de ce même groupe en utilisant astroalign.
    """
    # Define a local alias for the callback
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    # Internal GPU FFT phase-correlation aligner (translation only)
    def _fft_phase_shift(src2d: np.ndarray, ref2d: np.ndarray) -> tuple[int, int, float]:
        """Return (dy, dx, confidence_ratio). Uses CuPy if available, else NumPy."""
        # Use luminance-like for safety (2D already)
        if src2d.shape != ref2d.shape:
            return 0, 0, 0.0
        h, w = src2d.shape
        use_gpu = GPU_AVAILABLE
        try:
            if use_gpu:
                import cupy as cp  # type: ignore
                a = cp.asarray(src2d, dtype=cp.float32)
                b = cp.asarray(ref2d, dtype=cp.float32)
                # Remove DC to help stability
                a = a - cp.nanmean(a)
                b = b - cp.nanmean(b)
                Fa = cp.fft.fftn(a)
                Fb = cp.fft.fftn(b)
                R = Fa * cp.conj(Fb)
                denom = cp.maximum(cp.abs(R), 1e-12)
                Rn = R / denom
                r = cp.fft.ifftn(Rn).real
                peak_val = float(cp.max(r))
                peak_idx = tuple(int(x) for x in cp.unravel_index(int(cp.argmax(r)), r.shape))
                med_val = float(cp.median(r)) if r.size > 0 else 0.0
                py, px = peak_idx
                dy = py if py <= h // 2 else py - h
                dx = px if px <= w // 2 else px - w
                conf = (peak_val / max(1e-6, med_val + 1e-6)) if np.isfinite(med_val) else (peak_val / 1e-6)
                return int(dy), int(dx), float(conf)
            else:
                a = src2d.astype(np.float32) - np.nanmean(src2d.astype(np.float32))
                b = ref2d.astype(np.float32) - np.nanmean(ref2d.astype(np.float32))
                Fa = np.fft.fftn(a)
                Fb = np.fft.fftn(b)
                R = Fa * np.conj(Fb)
                denom = np.maximum(np.abs(R), 1e-12)
                Rn = R / denom
                r = np.fft.ifftn(Rn).real
                peak_val = float(np.max(r))
                py, px = np.unravel_index(int(np.argmax(r)), r.shape)
                med_val = float(np.median(r)) if r.size > 0 else 0.0
                dy = py if py <= h // 2 else py - h
                dx = px if px <= w // 2 else px - w
                conf = (peak_val / max(1e-6, med_val + 1e-6)) if np.isfinite(med_val) else (peak_val / 1e-6)
                return int(dy), int(dx), float(conf)
        except Exception:
            return 0, 0, 0.0

    def _apply_integer_shift_hw_or_hwc(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
        if dy == 0 and dx == 0:
            return img.copy()
        h, w = img.shape[:2]
        out = np.zeros_like(img, dtype=np.float32)
        ys = slice(max(0, dy), min(h, h + dy))
        yt = slice(max(0, -dy), max(0, -dy) + (ys.stop - ys.start))
        xs = slice(max(0, dx), min(w, w + dx))
        xt = slice(max(0, -dx), max(0, -dx) + (xs.stop - xs.start))
        if img.ndim == 2:
            out[yt, xt] = img[ys, xs]
        else:
            out[yt, xt, :] = img[ys, xs, :]
        return out

    if not ASTROALIGN_AVAILABLE or astroalign_module is None:
        _pcb("aligngroup_error_astroalign_unavailable", lvl="WARN")
        # We'll still try FFT phase-correlation if possible
        if not image_data_list or not (0 <= reference_image_index < len(image_data_list)):
            empty = [None] * len(image_data_list)
            return empty, list(range(len(empty)))
        ref = image_data_list[reference_image_index]
        if ref is None:
            empty = [None] * len(image_data_list)
            return empty, list(range(len(empty)))
        if ref.ndim == 3 and ref.shape[-1] == 3:
            ref_lum = 0.299 * ref[..., 0] + 0.587 * ref[..., 1] + 0.114 * ref[..., 2]
        else:
            ref_lum = ref.astype(np.float32, copy=False)
        aligned = [None] * len(image_data_list)
        for i, src in enumerate(image_data_list):
            if src is None:
                continue
            if i == reference_image_index:
                aligned[i] = ref.astype(np.float32, copy=True)
                continue
            src_lum = (0.299 * src[..., 0] + 0.587 * src[..., 1] + 0.114 * src[..., 2]).astype(np.float32) if (src.ndim == 3 and src.shape[-1] == 3) else src.astype(np.float32)
            dy, dx, conf = _fft_phase_shift(src_lum, ref_lum)
            if abs(dy) > 0 or abs(dx) > 0:
                aligned[i] = _apply_integer_shift_hw_or_hwc(src.astype(np.float32, copy=False), dy, dx)
            else:
                aligned[i] = src.astype(np.float32, copy=True)
        failed_idx = [idx for idx, img in enumerate(aligned) if img is None]
        return aligned, failed_idx

    if not image_data_list or not (0 <= reference_image_index < len(image_data_list)):
        _pcb("aligngroup_error_invalid_input_list_or_ref_index", lvl="ERROR", ref_idx=reference_image_index)
        empty = [None] * len(image_data_list)
        return empty, list(range(len(empty)))

    reference_image_adu = image_data_list[reference_image_index]
    if reference_image_adu is None:
        _pcb("aligngroup_error_ref_image_none", lvl="ERROR", ref_idx=reference_image_index)
        empty = [None] * len(image_data_list)
        return empty, list(range(len(empty)))
    
    if reference_image_adu.dtype != np.float32:
        _pcb(f"AlignGroup: Image de référence (index {reference_image_index}) convertie en float32.", lvl="DEBUG_DETAIL")
        reference_image_adu = reference_image_adu.astype(np.float32)

    _pcb(f"AlignGroup: Alignement intra-tuile sur réf. idx {reference_image_index} (shape {reference_image_adu.shape}).", lvl="DEBUG")
    aligned_images = [None] * len(image_data_list)

    for i, source_image_adu_orig in enumerate(image_data_list):
        if source_image_adu_orig is None:
            _pcb(f"AlignGroup: Image source {i} est None, ignorée.", lvl="WARN")
            continue

        source_image_adu = source_image_adu_orig.astype(np.float32, copy=False) 

        if i == reference_image_index:
            aligned_images[i] = reference_image_adu.copy() 
            _pcb(f"AlignGroup: Image {i} est la référence, copiée.", lvl="DEBUG_DETAIL")
            continue

        _pcb(f"AlignGroup: Alignement image {i} (shape {source_image_adu.shape}) sur référence...", lvl="DEBUG_DETAIL")
        try:
            # Try GPU FFT phase correlation first for robust global translation
            try_fft = True
            if try_fft:
                src_lum = (0.299 * source_image_adu[..., 0] + 0.587 * source_image_adu[..., 1] + 0.114 * source_image_adu[..., 2]).astype(np.float32) if (source_image_adu.ndim == 3 and source_image_adu.shape[-1] == 3) else source_image_adu
                ref_lum = (0.299 * reference_image_adu[..., 0] + 0.587 * reference_image_adu[..., 1] + 0.114 * reference_image_adu[..., 2]).astype(np.float32) if (reference_image_adu.ndim == 3 and reference_image_adu.shape[-1] == 3) else reference_image_adu
                dy, dx, conf = _fft_phase_shift(src_lum, ref_lum)
                if abs(dy) + abs(dx) > 0 and conf >= 3.0:  # heuristic confidence
                    aligned_images[i] = _apply_integer_shift_hw_or_hwc(source_image_adu, dy, dx)
                    _pcb(f"AlignGroup: FFT shift applied (dy={dy}, dx={dx}, conf={conf:.2f}).", lvl="DEBUG_DETAIL")
                    continue
            # Fall back to astroalign for fine/affine alignment
            # Garantir des buffers writables/contigus pour astroalign afin d'éviter
            # "ValueError: buffer source array is read-only" avec des memmaps read-only
            src_for_aa = (
                source_image_adu if (getattr(source_image_adu, 'flags', None)
                                     and source_image_adu.flags.writeable
                                     and source_image_adu.flags.c_contiguous)
                else np.array(source_image_adu, dtype=np.float32, copy=True, order='C')
            )
            ref_for_aa = (
                reference_image_adu if (getattr(reference_image_adu, 'flags', None)
                                        and reference_image_adu.flags.writeable
                                        and reference_image_adu.flags.c_contiguous)
                else np.array(reference_image_adu, dtype=np.float32, copy=True, order='C')
            )
            aligned_image_output, footprint_mask = astroalign_module.register(
                source=src_for_aa, target=ref_for_aa,
                detection_sigma=detection_sigma, min_area=min_area,
                propagate_mask=propagate_mask
            )
            if aligned_image_output is not None:
                if aligned_image_output.shape != reference_image_adu.shape:
                    _pcb("aligngroup_warn_shape_mismatch_after_align", lvl="WARN", img_idx=i, 
                              aligned_shape=aligned_image_output.shape, ref_shape=reference_image_adu.shape)
                    aligned_images[i] = None
                else:
                    aligned_images[i] = aligned_image_output.astype(np.float32)
                    _pcb(f"AlignGroup: Image {i} alignée.", lvl="DEBUG_DETAIL")
            else:
                _pcb("aligngroup_warn_register_returned_none", lvl="WARN", img_idx=i)
                aligned_images[i] = None
        except astroalign_module.MaxIterError:
            _pcb("aligngroup_warn_max_iter_error", lvl="WARN", img_idx=i)
            aligned_images[i] = None
        except ValueError as ve:
            _pcb("aligngroup_warn_value_error", lvl="WARN", img_idx=i, error=str(ve))
            aligned_images[i] = None
        except Exception as e_align:
            _pcb("aligngroup_error_exception_aligning", lvl="ERROR", img_idx=i, error_type=type(e_align).__name__, error_msg=str(e_align))
            _pcb(f"AlignGroup Traceback: {traceback.format_exc()}", lvl="DEBUG_DETAIL")
            aligned_images[i] = None
    failed_indices = [idx for idx, img in enumerate(aligned_images) if img is None]
    return aligned_images, failed_indices



def _normalize_images_linear_fit(image_list_hwc_float32: list[np.ndarray],
                                 reference_index: int = 0,
                                 low_percentile: float = 25.0,
                                 high_percentile: float = 90.0,
                                 progress_callback: callable = None,
                                 use_gpu: bool = False):
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")
    _pcb("norm_linear_fit_starting", lvl="DEBUG", num_images=len(image_list_hwc_float32), ref_idx=reference_index, low_p=low_percentile, high_p=high_percentile)
    if not image_list_hwc_float32:
        _pcb("norm_linear_fit_error_no_images", lvl="WARN"); return []
    if not (0 <= reference_index < len(image_list_hwc_float32) and image_list_hwc_float32[reference_index] is not None):
        _pcb("norm_linear_fit_error_invalid_ref_index_or_ref_none", lvl="ERROR", ref_idx=reference_index)
        return [img.copy() if img is not None else None for img in image_list_hwc_float32]

    ref_image_hwc_float32 = image_list_hwc_float32[reference_index]
    if ref_image_hwc_float32.dtype != np.float32:
        ref_image_hwc_float32 = ref_image_hwc_float32.astype(np.float32, copy=False)
    is_color = ref_image_hwc_float32.ndim == 3 and ref_image_hwc_float32.shape[-1] == 3
    num_channels = 3 if is_color else 1
    normalized_image_list = [None] * len(image_list_hwc_float32)
    ref_stats_per_channel = []
    for c_idx_ref in range(num_channels):
        ref_channel_2d = ref_image_hwc_float32[..., c_idx_ref] if is_color else ref_image_hwc_float32
        ref_low, ref_high = _calculate_robust_stats_for_linear_fit(ref_channel_2d, low_percentile, high_percentile, progress_callback, use_gpu=use_gpu)
        ref_stats_per_channel.append((ref_low, ref_high))
        _pcb(f"NormLinFit: Réf. Canal {c_idx_ref}: StatLow={ref_low:.3g}, StatHigh={ref_high:.3g}", lvl="DEBUG_DETAIL")
        if abs(ref_high - ref_low) < 1e-5:
             _pcb(f"NormLinFit: AVERT Réf. Canal {c_idx_ref} est (presque) plat.", lvl="WARN")

    for i, src_image_hwc_orig_float32 in enumerate(image_list_hwc_float32):
        if src_image_hwc_orig_float32 is None: continue
        if i == reference_index:
            normalized_image_list[i] = ref_image_hwc_float32.copy()
            continue
        src_image_hwc_float32 = src_image_hwc_orig_float32
        if src_image_hwc_float32.dtype != np.float32:
            src_image_hwc_float32 = src_image_hwc_float32.astype(np.float32, copy=True)
        else:
            src_image_hwc_float32 = src_image_hwc_float32.copy()
        if src_image_hwc_float32.shape != ref_image_hwc_float32.shape:
            _pcb(f"NormLinFit: AVERT Img {i} shape {src_image_hwc_float32.shape} != réf {ref_image_hwc_float32.shape}. Ignorée.", lvl="WARN")
            continue
        for c_idx_src in range(num_channels):
            src_channel_2d = src_image_hwc_float32[..., c_idx_src] if is_color else src_image_hwc_float32
            ref_low, ref_high = ref_stats_per_channel[c_idx_src]
            src_low, src_high = _calculate_robust_stats_for_linear_fit(src_channel_2d, low_percentile, high_percentile, progress_callback, use_gpu=use_gpu)
            a = 1.0; b = 0.0
            delta_src = src_high - src_low; delta_ref = ref_high - ref_low
            if abs(delta_src) > 1e-5:
                if abs(delta_ref) > 1e-5: a = delta_ref / delta_src; b = ref_low - a * src_low
                else: b = ref_low - src_low # a=1
            else:
                if abs(delta_ref) > 1e-5: a = 0.0; b = ref_low
                else: b = ref_low - src_low # a=1
            if abs(a - 1.0) > 1e-3 or abs(b) > 1e-3 * max(abs(ref_low), abs(src_low), 1.0):
                 _pcb(f"NormLinFit: Img {i} C{c_idx_src}: Src(L/H)=({src_low:.3g}/{src_high:.3g}) -> Coeffs a={a:.3f}, b={b:.3f}", lvl="DEBUG_DETAIL")
            transformed_channel = a * src_channel_2d + b
            if is_color: src_image_hwc_float32[..., c_idx_src] = transformed_channel
            else: src_image_hwc_float32 = transformed_channel
        normalized_image_list[i] = src_image_hwc_float32
    _pcb("norm_linear_fit_finished", lvl="DEBUG", num_normalized_successfully=sum(1 for img in normalized_image_list if img is not None))
    return normalized_image_list



# Dans zemosaic_align_stack.py

def _normalize_images_sky_mean(image_list: list[np.ndarray | None], 
                               reference_index: int = 0,
                               sky_percentile: float = 25.0, # Percentile pour estimer le fond de ciel
                               progress_callback: callable = None,
                               use_gpu: bool = False) -> list[np.ndarray | None]:
    """
    Normalise une liste d'images en ajustant leur fond de ciel moyen (estimé par percentile)
    pour correspondre à celui de l'image de référence.
    Opère sur la luminance pour les images couleur.
    """
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    if not image_list:
        _pcb("norm_skymean_error_no_images", lvl="WARN")
        return []
    
    if not (0 <= reference_index < len(image_list) and image_list[reference_index] is not None):
        _pcb("norm_skymean_error_invalid_ref", lvl="ERROR", ref_idx=reference_index)
        # Retourner une copie des images originales si la référence est invalide
        return [img.copy() if img is not None else None for img in image_list]

    _pcb(f"NormSkyMean: Début normalisation par fond de ciel (percentile {sky_percentile}%) sur réf. idx {reference_index}.", lvl="DEBUG")
    
    ref_image_adu = image_list[reference_index]
    # S'assurer que l'image de référence est en float32 pour les calculs
    if ref_image_adu.dtype != np.float32:
        ref_image_adu_float = ref_image_adu.astype(np.float32, copy=True)
    else:
        ref_image_adu_float = ref_image_adu # Pas besoin de copier si déjà float32 et on ne le modifie pas directement

    # --- Calculer le fond de ciel de référence ---
    ref_sky_level = None
    target_data_for_ref_sky = None
    if ref_image_adu_float.ndim == 3 and ref_image_adu_float.shape[-1] == 3: # Couleur HWC
        luminance_ref = 0.299 * ref_image_adu_float[..., 0] + \
                        0.587 * ref_image_adu_float[..., 1] + \
                        0.114 * ref_image_adu_float[..., 2]
        target_data_for_ref_sky = luminance_ref
    elif ref_image_adu_float.ndim == 2: # Monochrome HW
        target_data_for_ref_sky = ref_image_adu_float
    
    if target_data_for_ref_sky is not None and target_data_for_ref_sky.size > 0:
        try:
            # Utiliser nanpercentile pour être robuste aux NaNs potentiels
            if use_gpu and GPU_AVAILABLE:
                try:
                    ref_sky_level = float(
                        _gpu_nanpercentile(target_data_for_ref_sky, sky_percentile)
                    )
                except Exception:
                    ref_sky_level = float(
                        np.nanpercentile(target_data_for_ref_sky, sky_percentile)
                    )
            else:
                ref_sky_level = np.nanpercentile(target_data_for_ref_sky, sky_percentile)
            _pcb(f"NormSkyMean: Fond de ciel de référence (img idx {reference_index}) estimé à {ref_sky_level:.3g}", lvl="DEBUG_DETAIL")
        except Exception as e_perc_ref:
            _pcb(f"NormSkyMean: Erreur calcul percentile réf: {e_perc_ref}", lvl="WARN")
            # Si échec, on ne peut pas normaliser, retourner les images telles quelles (ou des copies)
            return [img.copy() if img is not None else None for img in image_list]
    else:
        _pcb("NormSkyMean: Impossible de déterminer les données pour le fond de ciel de référence.", lvl="WARN")
        return [img.copy() if img is not None else None for img in image_list]

    if ref_sky_level is None or not np.isfinite(ref_sky_level):
        _pcb(f"NormSkyMean: Fond de ciel de référence invalide ({ref_sky_level}). Normalisation annulée.", lvl="ERROR")
        return [img.copy() if img is not None else None for img in image_list]

    # --- Normaliser chaque image ---
    normalized_image_list = [None] * len(image_list)
    for i, current_image_adu in enumerate(image_list):
        if current_image_adu is None:
            normalized_image_list[i] = None
            continue

        # Faire une copie pour la modification, s'assurer qu'elle est float32
        if current_image_adu.dtype != np.float32:
            img_to_normalize_float = current_image_adu.astype(np.float32, copy=True)
        else:
            img_to_normalize_float = current_image_adu.copy() # Toujours copier pour modifier

        if i == reference_index:
            normalized_image_list[i] = img_to_normalize_float # C'est déjà la référence (ou sa copie float32)
            _pcb(f"NormSkyMean: Image {i} est la référence, copiée.", lvl="DEBUG_VERY_DETAIL")
            continue

        target_data_for_current_sky = None
        is_current_color = img_to_normalize_float.ndim == 3 and img_to_normalize_float.shape[-1] == 3
        if is_current_color:
            luminance_current = 0.299 * img_to_normalize_float[..., 0] + \
                                0.587 * img_to_normalize_float[..., 1] + \
                                0.114 * img_to_normalize_float[..., 2]
            target_data_for_current_sky = luminance_current
        elif img_to_normalize_float.ndim == 2:
            target_data_for_current_sky = img_to_normalize_float
        
        current_sky_level = None
        if target_data_for_current_sky is not None and target_data_for_current_sky.size > 0 :
            try:
                if use_gpu and GPU_AVAILABLE:
                    try:
                        current_sky_level = float(
                            _gpu_nanpercentile(target_data_for_current_sky, sky_percentile)
                        )
                    except Exception:
                        current_sky_level = float(
                            np.nanpercentile(target_data_for_current_sky, sky_percentile)
                        )
                else:
                    current_sky_level = np.nanpercentile(target_data_for_current_sky, sky_percentile)
            except Exception as e_perc_curr:
                 _pcb(f"NormSkyMean: Erreur calcul percentile image {i}: {e_perc_curr}. Image non normalisée.", lvl="WARN")
                 normalized_image_list[i] = img_to_normalize_float # Retourner la copie non modifiée
                 continue
        
        if current_sky_level is not None and np.isfinite(current_sky_level):
            offset = ref_sky_level - current_sky_level
            img_to_normalize_float += offset # Appliquer l'offset à tous les canaux si couleur, ou à l'image si mono
            normalized_image_list[i] = img_to_normalize_float
            _pcb(f"NormSkyMean: Image {i}, fond_ciel={current_sky_level:.3g}, offset_appliqué={offset:.3g}", lvl="DEBUG_VERY_DETAIL")
        else:
            _pcb(f"NormSkyMean: Fond de ciel invalide pour image {i} ({current_sky_level}). Image non normalisée.", lvl="WARN")
            normalized_image_list[i] = img_to_normalize_float # Retourner la copie non modifiée

    _pcb("NormSkyMean: Normalisation par fond de ciel terminée.", lvl="DEBUG")
    return normalized_image_list



def _calculate_image_weights_noise_variance(
    image_list: list[np.ndarray | None],
    progress_callback: callable = None,
) -> list[np.ndarray | None]:
    """
    Calcule des poids qualité basés sur l'inverse de la variance du bruit.
    Retourne des poids compacts par image: scalaire pour mono ou vecteur 3 canaux pour couleur.
    """
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    if not image_list:
        _pcb("weight_noisevar_error_no_images", lvl="WARN")
        return []

    if not SIGMA_CLIP_AVAILABLE or sigma_clipped_stats_func is None:
        _pcb("weight_noisevar_warn_astropy_stats_unavailable", lvl="WARN")
        weights = []
        for img in image_list:
            if img is None:
                weights.append(None)
            elif isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
                weights.append(np.asarray([1.0, 1.0, 1.0], dtype=np.float32))
            else:
                weights.append(np.asarray(1.0, dtype=np.float32))
        return weights

    # Va stocker pour chaque image valide:
    # - Pour les images couleur: une liste [var_R, var_G, var_B]
    # - Pour les images monochrome: une liste [var_Mono]
    variances_per_image_channels = [] 
    valid_image_indices = []

    _pcb(f"WeightNoiseVar: Début calcul des poids (par canal si couleur) pour {len(image_list)} images.", lvl="DEBUG")

    for i, image_data_adu in enumerate(image_list):
        if image_data_adu is None:
            continue # Sera géré à la fin
        
        if image_data_adu.size == 0:
            _pcb(f"WeightNoiseVar: Image {i} est vide, poids non calculé pour celle-ci.", lvl="WARN")
            continue

        img_for_stats = image_data_adu
        if image_data_adu.dtype != np.float32:
            # Il est crucial de faire une copie si on change le type pour ne pas modifier l'original dans image_list
            img_for_stats = image_data_adu.astype(np.float32, copy=True) 

        current_image_channel_variances = []
        num_channels_in_image = 0

        if img_for_stats.ndim == 3 and img_for_stats.shape[-1] == 3: # Image couleur HWC
            num_channels_in_image = img_for_stats.shape[-1]
            for c_idx in range(num_channels_in_image):
                channel_data = img_for_stats[..., c_idx]
                if channel_data.size == 0:
                    _pcb(f"WeightNoiseVar: Image {i}, Canal {c_idx} vide.", lvl="WARN")
                    current_image_channel_variances.append(np.inf) # Poids sera quasi nul
                    continue
                try:
                    # Utiliser sigma_lower, sigma_upper pour un écrêtage plus robuste
                    _, _, stddev_ch = sigma_clipped_stats_func(
                        channel_data, sigma_lower=3.0, sigma_upper=3.0, maxiters=5
                    )
                    if stddev_ch is not None and np.isfinite(stddev_ch) and stddev_ch > 1e-9: # 1e-9 pour éviter variance nulle
                        current_image_channel_variances.append(stddev_ch**2)
                    else:
                        _pcb(f"WeightNoiseVar: Image {i}, Canal {c_idx}, stddev invalide ({stddev_ch}). Variance Inf.", lvl="WARN")
                        current_image_channel_variances.append(np.inf)
                except Exception as e_stats_ch:
                    _pcb(f"WeightNoiseVar: Erreur stats image {i}, canal {c_idx}: {e_stats_ch}", lvl="WARN")
                    current_image_channel_variances.append(np.inf)
            
        elif img_for_stats.ndim == 2: # Image monochrome HW
            num_channels_in_image = 1 # Conceptuellement
            if img_for_stats.size == 0:
                _pcb(f"WeightNoiseVar: Image monochrome {i} vide.", lvl="WARN")
                current_image_channel_variances.append(np.inf)
            else:
                try:
                    _, _, stddev = sigma_clipped_stats_func(
                        img_for_stats, sigma_lower=3.0, sigma_upper=3.0, maxiters=5
                    )
                    if stddev is not None and np.isfinite(stddev) and stddev > 1e-9:
                        current_image_channel_variances.append(stddev**2)
                    else:
                        _pcb(f"WeightNoiseVar: Image monochrome {i}, stddev invalide ({stddev}). Variance Inf.", lvl="WARN")
                        current_image_channel_variances.append(np.inf)
                except Exception as e_stats_mono:
                    _pcb(f"WeightNoiseVar: Erreur stats image monochrome {i}: {e_stats_mono}", lvl="WARN")
                    current_image_channel_variances.append(np.inf)
        else:
            _pcb(f"WeightNoiseVar: Image {i} a une forme non supportée ({img_for_stats.shape}).", lvl="WARN")
            continue # Passe à l'image suivante
        
        # Si on a réussi à calculer des variances pour les canaux de cette image
        if len(current_image_channel_variances) == num_channels_in_image and num_channels_in_image > 0:
            variances_per_image_channels.append(current_image_channel_variances)
            valid_image_indices.append(i)
        elif num_channels_in_image > 0 : # Si on s'attendait à des canaux mais on n'a pas toutes les variances
             _pcb(f"WeightNoiseVar: N'a pas pu calculer toutes les variances de canal pour l'image {i}.", lvl="WARN")


    if not variances_per_image_channels:
        _pcb("weight_noisevar_warn_no_variances_calculated_at_all", lvl="WARN")
        return [np.ones_like(img, dtype=np.float32) if img is not None else None for img in image_list]

    all_finite_variances = []
    for var_list_for_img in variances_per_image_channels:
        for var_val in var_list_for_img:
            if np.isfinite(var_val) and var_val > 1e-18: # Seuil très bas pour variance valide
                all_finite_variances.append(var_val)
    
    min_overall_variance = np.min(all_finite_variances) if all_finite_variances else 1e-9
    if min_overall_variance <= 0: min_overall_variance = 1e-9 # Assurer qu'elle est positive

    _pcb(f"WeightNoiseVar: Variance minimale globale trouvée: {min_overall_variance:.3g}", lvl="DEBUG_DETAIL")

    output_weights_list = [None] * len(image_list)

    for idx_in_valid_arrays, original_image_idx in enumerate(valid_image_indices):
        original_img_data_shape_ref = image_list[original_image_idx]
        if original_img_data_shape_ref is None:
            continue

        variances_for_current_img = variances_per_image_channels[idx_in_valid_arrays]
        
        # Créer un poids compact pour cette image: (3,) pour couleur ou scalaire pour mono
        if original_img_data_shape_ref.ndim == 3 and len(variances_for_current_img) == original_img_data_shape_ref.shape[-1]:
            ch_weights = []
            for c_idx in range(original_img_data_shape_ref.shape[-1]):
                variance_ch = variances_for_current_img[c_idx]
                if np.isfinite(variance_ch) and variance_ch > 1e-18:
                    ch_weights.append(float(min_overall_variance / variance_ch))
                else:
                    ch_weights.append(1e-6)
            output_weights_list[original_image_idx] = np.asarray(ch_weights, dtype=np.float32)
        elif original_img_data_shape_ref.ndim == 2 and len(variances_for_current_img) == 1:
            variance_mono = variances_for_current_img[0]
            if np.isfinite(variance_mono) and variance_mono > 1e-18:
                calculated_weight = float(min_overall_variance / variance_mono)
            else:
                calculated_weight = 1e-6
            output_weights_list[original_image_idx] = np.asarray(calculated_weight, dtype=np.float32)

    # Pour les images qui n'ont pas pu être traitées (initialement None, ou erreur en cours de route)
    for i in range(len(image_list)):
        if output_weights_list[i] is None and image_list[i] is not None:
            _pcb("WeightNoiseVar: Image sans poids valide, fallback sur 1.0.", lvl="DEBUG_DETAIL")
            if isinstance(image_list[i], np.ndarray) and image_list[i].ndim == 3 and image_list[i].shape[-1] == 3:
                output_weights_list[i] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
            else:
                output_weights_list[i] = np.asarray(1.0, dtype=np.float32)
            
    num_actual_weights = sum(1 for w_arr in output_weights_list if w_arr is not None)
    _pcb(f"WeightNoiseVar: Calcul des poids (par canal si couleur) terminé. {num_actual_weights}/{len(image_list)} tableaux de poids retournés.", lvl="DEBUG")
    return output_weights_list



def _estimate_initial_fwhm(data_2d: np.ndarray, progress_callback: callable = None) -> float:
    """
    Tente d'estimer une FWHM initiale à partir des données 2D.
    Utilise la segmentation et les propriétés des sources.
    """
    _pcb_est = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    default_fwhm = 4.0 # Valeur de secours
    if data_2d.size < 1000: # Pas assez de données pour une estimation fiable
        _pcb_est("fwhm_est_data_insufficient", lvl="DEBUG_DETAIL", returned_fwhm=default_fwhm)
        return default_fwhm

    try:
        # Estimation simple du fond et du bruit pour la segmentation
        _, median, std = sigma_clipped_stats_func(data_2d, sigma=3.0, maxiters=5)
        if not (np.isfinite(median) and np.isfinite(std) and std > 1e-6):
            _pcb_est("fwhm_est_stats_invalid", lvl="DEBUG_DETAIL", returned_fwhm=default_fwhm)
            return default_fwhm
            
        threshold_seg_est = median + (3.0 * std) # Seuil pour la segmentation
        
        # Segmentation
        segm_map = detect_sources(data_2d, threshold_seg_est, npixels=5) # npixels minimum pour une source
        if segm_map is None:
            _pcb_est("fwhm_est_segmentation_failed", lvl="DEBUG_DETAIL", returned_fwhm=default_fwhm)
            return default_fwhm

        # Catalogue des sources
        cat = SourceCatalog(data_2d, segm_map)
        if not cat or len(cat) == 0:
            _pcb_est("fwhm_est_no_sources_in_catalog", lvl="DEBUG_DETAIL", returned_fwhm=default_fwhm)
            return default_fwhm

        fwhms_from_cat = []
        # On ne prend que les sources "raisonnables" pour l'estimation
        for props in cat:
            try:
                # equivalent_fwhm est une bonne estimation si la source est ~gaussienne
                # On filtre sur l'ellipticité pour ne garder que les sources rondes
                if props.eccentricity is not None and props.eccentricity < 0.5 and \
                   props.equivalent_fwhm is not None and np.isfinite(props.equivalent_fwhm) and \
                   1.0 < props.equivalent_fwhm < 20.0: # FWHM doit être dans une plage plausible
                    fwhms_from_cat.append(props.equivalent_fwhm.value)
            except AttributeError: # Certaines propriétés peuvent manquer
                continue
            if len(fwhms_from_cat) >= 100: # Limiter le nombre de sources pour l'estimation
                break
        
        if not fwhms_from_cat:
            _pcb_est("fwhm_est_no_valid_fwhm_from_cat", lvl="DEBUG_DETAIL", returned_fwhm=default_fwhm)
            return default_fwhm
            
        estimated_fwhm = np.nanmedian(fwhms_from_cat)
        if np.isfinite(estimated_fwhm) and 1.0 < estimated_fwhm < 15.0:
            _pcb_est("fwhm_est_success", lvl="DEBUG_DETAIL", estimated_fwhm=float(estimated_fwhm))
            return float(estimated_fwhm)
        else:
            _pcb_est("fwhm_est_median_invalid", lvl="DEBUG_DETAIL", median_fwhm=estimated_fwhm, returned_fwhm=default_fwhm)
            return default_fwhm

    except Exception as e_est:
        _pcb_est("fwhm_est_exception", lvl="WARN", error=str(e_est), returned_fwhm=default_fwhm)
        return default_fwhm


def _calculate_image_weights_noise_fwhm(
    image_list: list[np.ndarray | None],
    progress_callback: callable = None,
) -> list[np.ndarray | None]:
    """
    Calcule des poids basés sur l'inverse de la FWHM. Retourne des poids compacts
    par image: scalaire pour mono ou vecteur 3 canaux pour couleur.
    """
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    if not image_list:
        _pcb("weight_fwhm_error_no_images", lvl="WARN")
        return []

    if not PHOTOUTILS_AVAILABLE or not SIGMA_CLIP_AVAILABLE:
        missing_fwhm_deps = []
        if not PHOTOUTILS_AVAILABLE:
            missing_fwhm_deps.append("Photutils")
        if not SIGMA_CLIP_AVAILABLE:
            missing_fwhm_deps.append("Astropy.stats (SigmaClip)")
        _pcb("weight_fwhm_warn_deps_unavailable", lvl="WARN", missing=", ".join(missing_fwhm_deps))
        compact = []
        for img in image_list:
            if img is None:
                compact.append(None)
            elif isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
                compact.append(np.asarray([1.0, 1.0, 1.0], dtype=np.float32))
            else:
                compact.append(np.asarray(1.0, dtype=np.float32))
        return compact

    fwhm_values_per_image = [] 
    valid_image_indices_fwhm = []

    _pcb(f"WeightFWHM: Début calcul des poids FWHM pour {len(image_list)} images.", lvl="DEBUG")

    for i, image_data_adu in enumerate(image_list):
        if image_data_adu is None:
            continue
        
        if image_data_adu.size == 0:
            _pcb("weight_fwhm_img_empty", lvl="WARN", img_idx=i)
            continue

        img_for_fwhm_calc = image_data_adu
        if image_data_adu.dtype != np.float32:
            img_for_fwhm_calc = image_data_adu.astype(np.float32, copy=True)

        target_data_for_fwhm = None
        if img_for_fwhm_calc.ndim == 3 and img_for_fwhm_calc.shape[-1] == 3:
            luminance = 0.299 * img_for_fwhm_calc[..., 0] + \
                        0.587 * img_for_fwhm_calc[..., 1] + \
                        0.114 * img_for_fwhm_calc[..., 2]
            target_data_for_fwhm = luminance
        elif img_for_fwhm_calc.ndim == 2:
            target_data_for_fwhm = img_for_fwhm_calc
        else:
            _pcb("weight_fwhm_unsupported_shape", lvl="WARN", img_idx=i, shape=img_for_fwhm_calc.shape)
            continue
        
        if target_data_for_fwhm is None or target_data_for_fwhm.size < 50*50: # Besoin d'une taille minimale
             _pcb("weight_fwhm_insufficient_data", lvl="WARN", img_idx=i)
             continue
        
        try:
            estimated_initial_fwhm = _estimate_initial_fwhm(target_data_for_fwhm, progress_callback)
            _pcb(f"WeightFWHM: Image {i}, FWHM initiale estimée pour détection: {estimated_initial_fwhm:.2f} px", lvl="DEBUG_DETAIL")

            box_size_bg = min(target_data_for_fwhm.shape[0] // 8, target_data_for_fwhm.shape[1] // 8, 50)
            box_size_bg = max(box_size_bg, 16)
            
            sigma_clip_bg_obj = SigmaClip(sigma=3.0) # Renommé pour éviter conflit
            bkg_estimator_obj = MedianBackground()   # Renommé pour éviter conflit
            
            if not np.any(np.isfinite(target_data_for_fwhm)):
                _pcb("weight_fwhm_no_finite_data", lvl="WARN", img_idx=i)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue
            
            std_data_check = np.nanstd(target_data_for_fwhm)
            if std_data_check < 1e-6 :
                 _pcb("weight_fwhm_image_flat", lvl="DEBUG_DETAIL", img_idx=i, stddev=std_data_check)
                 fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue

            bkg_obj = None # Pour vérifier si bkg a été défini
            try:
                bkg_obj = Background2D(target_data_for_fwhm, (box_size_bg, box_size_bg), 
                                   filter_size=(3, 3), sigma_clip=sigma_clip_bg_obj, bkg_estimator=bkg_estimator_obj)
                data_subtracted = target_data_for_fwhm - bkg_obj.background
                threshold_daofind_val = 5.0 * bkg_obj.background_rms 
            except (ValueError, TypeError) as ve_bkg: 
                _pcb("weight_fwhm_bkg2d_error", lvl="WARN", img_idx=i, error=str(ve_bkg))
                _, median_glob, stddev_glob = sigma_clipped_stats_func(target_data_for_fwhm, sigma=3.0, maxiters=5)
                if not (np.isfinite(median_glob) and np.isfinite(stddev_glob)):
                    _pcb("weight_fwhm_global_stats_invalid", lvl="WARN", img_idx=i)
                    fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue
                data_subtracted = target_data_for_fwhm - median_glob
                threshold_daofind_val = 5.0 * stddev_glob

            # S'assurer que threshold_daofind_val est un scalaire positif
            if hasattr(threshold_daofind_val, 'mean'): threshold_daofind_val = np.abs(np.mean(threshold_daofind_val))
            else: threshold_daofind_val = np.abs(threshold_daofind_val)
            if threshold_daofind_val < 1e-5 : threshold_daofind_val = 1e-5 # Minimum seuil

            sources_table = None
            try:
                daofind_obj = DAOStarFinder(fwhm=estimated_initial_fwhm, threshold=threshold_daofind_val,
                                        sharplo=0.2, sharphi=1.0, roundlo=-0.8, roundhi=0.8, sky=0.0)
                sources_table = daofind_obj(data_subtracted)
            except Exception as e_daofind:
                 _pcb("weight_fwhm_daofind_error", lvl="WARN", img_idx=i, error=str(e_daofind))
                 fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue

            if sources_table is None or len(sources_table) < 5:
                _pcb("weight_fwhm_not_enough_sources_daofind", lvl="DEBUG_DETAIL", img_idx=i, count=len(sources_table) if sources_table is not None else 0)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue

            # Utilisation de SourceCatalog pour les propriétés morphologiques
            threshold_seg_val = 1.5 * (bkg_obj.background_rms if bkg_obj and hasattr(bkg_obj, 'background_rms') else np.nanstd(data_subtracted))
            if hasattr(threshold_seg_val, 'mean'): threshold_seg_val = np.abs(np.mean(threshold_seg_val))
            else: threshold_seg_val = np.abs(threshold_seg_val)
            if threshold_seg_val < 1e-5 : threshold_seg_val = 1e-5

            segm_map_cat = detect_sources(data_subtracted, threshold_seg_val, npixels=7) # npixels un peu plus grand
            if segm_map_cat is None:
                _pcb("weight_fwhm_segmentation_cat_failed", lvl="DEBUG_DETAIL", img_idx=i)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue
            
            # Filtrer les sources de DAOStarFinder avant de les passer à SourceCatalog
            h_img_cat, w_img_cat = data_subtracted.shape
            border_margin_cat = int(estimated_initial_fwhm * 2) # Marge basée sur FWHM
            
            # Assurer que les colonnes existent avant de filtrer
            cols_to_check = ['xcentroid', 'ycentroid', 'flux', 'sharpness', 'roundness1', 'roundness2']
            if not all(col in sources_table.colnames for col in cols_to_check):
                _pcb("weight_fwhm_missing_daofind_cols", lvl="WARN", img_idx=i, missing_cols=[c for c in cols_to_check if c not in sources_table.colnames])
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue


            valid_sources_mask_cat = (
                (sources_table['xcentroid'] > border_margin_cat) &
                (sources_table['xcentroid'] < w_img_cat - border_margin_cat) &
                (sources_table['ycentroid'] > border_margin_cat) &
                (sources_table['ycentroid'] < h_img_cat - border_margin_cat) &
                (sources_table['sharpness'] > 0.3) & (sources_table['sharpness'] < 0.95) & # Sources nettes mais pas trop
                (np.abs(sources_table['roundness1']) < 0.3) & (np.abs(sources_table['roundness2']) < 0.3) # Assez rondes
            )
            filtered_sources_table = sources_table[valid_sources_mask_cat]
            
            if not filtered_sources_table or len(filtered_sources_table) < 3:
                _pcb("weight_fwhm_not_enough_sources_after_filter_dao", lvl="DEBUG_DETAIL", img_idx=i)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue

            # Trier par flux et prendre les N plus brillantes
            filtered_sources_table.sort('flux', reverse=True)
            top_sources_table = filtered_sources_table[:100] # Limiter aux 100 plus brillantes
            
            # Passer les positions des sources détectées par DAOStarFinder à SourceCatalog
            try:
                cat_obj = SourceCatalog(data_subtracted, segm_map_cat, sources=top_sources_table)
            except Exception as e_scat: # SourceCatalog peut échouer si segm_map_cat est incompatible avec sources
                 _pcb("weight_fwhm_sourcecatalog_init_error", lvl="WARN", img_idx=i, error=str(e_scat))
                 fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue


            if not cat_obj or len(cat_obj) == 0:
                _pcb("weight_fwhm_no_sources_in_final_catalog", lvl="DEBUG_DETAIL", img_idx=i)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue

            fwhms_this_image = []
            for source_props in cat_obj:
                try:
                    # equivalent_fwhm est disponible et généralement fiable pour les sources bien segmentées.
                    # On pourrait aussi utiliser (semimajor_axis_sigma + semiminor_axis_sigma) / 2 * gaussian_sigma_to_fwhm
                    fwhm_val = source_props.equivalent_fwhm # C'est déjà une FWHM en pixels
                    if fwhm_val is not None and np.isfinite(fwhm_val) and \
                       0.8 < fwhm_val < (estimated_initial_fwhm * 2.5): # Doit être dans une plage raisonnable
                        fwhms_this_image.append(fwhm_val)
                except AttributeError: continue
                except Exception: continue
            
            if not fwhms_this_image:
                _pcb("weight_fwhm_no_valid_fwhm_from_catalog_props", lvl="DEBUG_DETAIL", img_idx=i)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i); continue

            median_fwhm_val = np.nanmedian(fwhms_this_image)
            if np.isfinite(median_fwhm_val) and 0.7 < median_fwhm_val < 20.0: # FWHM doit être > ~0.7 pixel et < 20
                fwhm_values_per_image.append(median_fwhm_val)
                valid_image_indices_fwhm.append(i)
                _pcb("weight_fwhm_success", lvl="DEBUG_DETAIL", img_idx=i, median_fwhm=median_fwhm_val, num_stars=len(fwhms_this_image))
            else:
                _pcb("weight_fwhm_median_fwhm_invalid", lvl="WARN", img_idx=i, median_fwhm=median_fwhm_val)
                fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i)

        except Exception as e_fwhm_main_loop:
            _pcb("weight_fwhm_mainloop_exception", lvl="ERROR", img_idx=i, error=str(e_fwhm_main_loop))
            _internal_logger.error(f"Traceback FWHM image {i}:", exc_info=True)
            fwhm_values_per_image.append(np.inf); valid_image_indices_fwhm.append(i)


    # --- Fin de la boucle sur les images ---

    if not fwhm_values_per_image: # Si aucune FWHM n'a pu être calculée pour aucune image
        _pcb("weight_fwhm_warn_no_fwhm_values_overall", lvl="WARN")
        return [np.ones_like(img, dtype=np.float32) if img is not None else None for img in image_list]

    finite_fwhms_all = [f for f in fwhm_values_per_image if np.isfinite(f) and f > 0.1] # 0.1 seuil très bas
    if not finite_fwhms_all:
        _pcb("weight_fwhm_warn_all_fwhm_are_infinite", lvl="WARN")
        return [np.ones_like(img, dtype=np.float32) if img is not None else None for img in image_list]

    min_overall_valid_fwhm = np.min(finite_fwhms_all)
    if min_overall_valid_fwhm < 0.5 : min_overall_valid_fwhm = 0.5 # FWHM minimale raisonnable

    _pcb(f"WeightFWHM: FWHM minimale globale valide: {min_overall_valid_fwhm:.2f} px", lvl="DEBUG_DETAIL")

    final_calculated_weights_scalar_fwhm = {}
    for idx_in_valid_list, original_idx in enumerate(valid_image_indices_fwhm):
        fwhm_current_image = fwhm_values_per_image[idx_in_valid_list]
        weight_val = 1e-6 # Poids par défaut très faible
        if np.isfinite(fwhm_current_image) and fwhm_current_image > 0.1:
            # Poids = (min_FWHM / FWHM_image) ^ N. Ici N=1.
            # Cela donne un poids de 1 à la meilleure image, <1 aux autres.
            # Si FWHM_image est plus petit que min_overall_valid_fwhm (ne devrait pas arriver), clamp à 1.
            weight_val = min_overall_valid_fwhm / max(fwhm_current_image, min_overall_valid_fwhm)
        final_calculated_weights_scalar_fwhm[original_idx] = weight_val
        _pcb(f"WeightFWHM: Img idx_orig={original_idx}, FWHM={fwhm_current_image:.2f}, PoidsRelFinal={weight_val:.3f}", lvl="DEBUG_DETAIL")

    output_weights_list_fwhm = [None] * len(image_list)
    for i, original_image_data in enumerate(image_list):
        if original_image_data is None:
            output_weights_list_fwhm[i] = None
        elif i in final_calculated_weights_scalar_fwhm:
            scalar_w_fwhm = float(final_calculated_weights_scalar_fwhm[i])
            if isinstance(original_image_data, np.ndarray) and original_image_data.ndim == 3 and original_image_data.shape[-1] == 3:
                output_weights_list_fwhm[i] = np.asarray([scalar_w_fwhm] * 3, dtype=np.float32)
            else:
                output_weights_list_fwhm[i] = np.asarray(scalar_w_fwhm, dtype=np.float32)
        else:
            _pcb("weight_fwhm_fallback_weight_one", lvl="DEBUG_DETAIL", img_idx=i)
            if isinstance(original_image_data, np.ndarray) and original_image_data.ndim == 3 and original_image_data.shape[-1] == 3:
                output_weights_list_fwhm[i] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
            else:
                output_weights_list_fwhm[i] = np.asarray(1.0, dtype=np.float32)
            
    num_actual_weights_fwhm = sum(1 for w_arr in output_weights_list_fwhm if w_arr is not None)
    _pcb(f"WeightFWHM: Calcul des poids FWHM terminé. {num_actual_weights_fwhm}/{len(image_list)} tableaux de poids retournés.", lvl="DEBUG")
    return output_weights_list_fwhm


def _compute_quality_weights(
    image_list: list[np.ndarray | None],
    requested_method: str,
    progress_callback: Callable | None = None,
) -> tuple[list[np.ndarray | None] | None, str, dict[str, float] | None]:
    """Compute per-frame quality weights according to the requested method.

    Returns compact per-frame weights to avoid materializing full (H,W[,C]) planes:
    - For color: arrays of shape (1, 1, 3)
    - For mono: arrays of shape (1,)
    """

    if not image_list:
        return None, "none", None

    method_norm = str(requested_method or "none").lower().strip()
    if not method_norm or method_norm == "none":
        return None, "none", None

    effective_method = method_norm

    if method_norm == "noise_variance":
        weights_source = _calculate_image_weights_noise_variance(
            image_list, progress_callback=progress_callback
        )
    elif method_norm == "noise_fwhm":
        if not PHOTOUTILS_AVAILABLE:
            _log_stack_message(
                "[Stack] Weighting 'noise_fwhm' requested but Photutils unavailable -> fallback to 'noise_variance'.",
                "WARN",
                progress_callback,
            )
            effective_method = "noise_variance"
            weights_source = _calculate_image_weights_noise_variance(
                image_list, progress_callback=progress_callback
            )
        else:
            weights_source = _calculate_image_weights_noise_fwhm(
                image_list, progress_callback=progress_callback
            )
            has_data = bool(weights_source and any(w is not None for w in weights_source))
            if not has_data:
                _log_stack_message(
                    "[Stack] FWHM weighting produced no usable weights; falling back to 'noise_variance'.",
                    "WARN",
                    progress_callback,
                )
                effective_method = "noise_variance"
                weights_source = _calculate_image_weights_noise_variance(
                    image_list, progress_callback=progress_callback
                )
    else:
        _log_stack_message(
            f"[Stack] Unknown weighting method '{method_norm}' requested; ignoring weights.",
            "WARN",
            progress_callback,
        )
        return None, "none", None

    if not weights_source:
        return None, "none", None

    sanitized: list[np.ndarray | None] = [None] * len(image_list)
    min_val: float | None = None
    max_val: float | None = None
    effective_frames = 0

    for idx, frame in enumerate(image_list):
        if frame is None:
            continue
        frame_arr = np.asarray(frame, dtype=np.float32)
        base_weight = weights_source[idx] if idx < len(weights_source) else None
        if base_weight is None:
            continue
        w = np.asarray(base_weight, dtype=np.float32, copy=False)
        # Compactify weight to scalar or per-channel 3-vector
        if frame_arr.ndim == 3 and frame_arr.shape[-1] == 3:
            # Color: expect per-channel weight
            if w.ndim == 0:
                w_compact = np.full((1, 1, 3), float(w), dtype=np.float32)
            elif w.ndim == 1 and w.shape[0] == 3:
                w_compact = w.reshape((1, 1, 3)).astype(np.float32, copy=False)
            else:
                try:
                    chv = np.nanmean(w, axis=(0, 1)) if w.ndim >= 2 else np.asarray(w)
                    chv = np.asarray(chv, dtype=np.float32).reshape((1, 1, -1))
                    if chv.shape[-1] == 3:
                        w_compact = chv
                    else:
                        w_compact = np.ones((1, 1, 3), dtype=np.float32)
                except Exception:
                    w_compact = np.ones((1, 1, 3), dtype=np.float32)
        else:
            # Mono: scalar
            try:
                val = float(np.nanmean(w)) if w.ndim > 0 else float(w)
            except Exception:
                val = 1.0
            w_compact = np.asarray([val], dtype=np.float32)

        if not np.any(np.isfinite(w_compact)):
            continue
        arr_min = float(np.nanmin(w_compact))
        arr_max = float(np.nanmax(w_compact))
        has_effect = (
            abs(arr_max - arr_min) >= 1e-6
            or abs(arr_min - 1.0) >= 1e-6
            or abs(arr_max - 1.0) >= 1e-6
        )
        if not has_effect:
            continue
        sanitized[idx] = w_compact
        effective_frames += 1
        if np.isfinite(arr_min):
            min_val = arr_min if min_val is None else min(min_val, arr_min)
        if np.isfinite(arr_max):
            max_val = arr_max if max_val is None else max(max_val, arr_max)

    if effective_frames == 0 or min_val is None or max_val is None:
        if effective_method != "none":
            _log_stack_message(
                f"[Stack] Weighting method '{effective_method}' produced uniform weights; continuing without weighting.",
                "WARN",
                progress_callback,
            )
        return None, "none", None

    stats = {"min": float(min_val), "max": float(max_val), "frames": effective_frames}
    return sanitized, effective_method, stats

def _reject_outliers_kappa_sigma(stacked_array_NHDWC, sigma_low, sigma_high, progress_callback=None):
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")
    
    _pcb(f"RejKappaSigma: Rejet Kappa-Sigma (low={sigma_low}, high={sigma_high}).", lvl="DEBUG")
    if not SIGMA_CLIP_AVAILABLE or sigma_clipped_stats_func is None:
        _pcb("stackhelper_warn_kappa_sigma_astropy_unavailable", lvl="WARN")
        return stacked_array_NHDWC, np.ones_like(stacked_array_NHDWC, dtype=bool) 
    
    rejection_mask = np.ones_like(stacked_array_NHDWC, dtype=bool)
    output_data_with_nans = stacked_array_NHDWC.copy()

    if stacked_array_NHDWC.ndim == 4: # Couleur (N, H, W, C)
        total_steps = stacked_array_NHDWC.shape[3]
        start_time = time.time()
        last_time = start_time
        step_times = []
        for c in range(total_steps):
            channel_data = stacked_array_NHDWC[..., c]
            try: 
                _, median_ch, stddev_ch = sigma_clipped_stats_func(channel_data, sigma_lower=sigma_low, sigma_upper=sigma_high, axis=0, maxiters=5)
            except TypeError: 
                _, median_ch, stddev_ch = sigma_clipped_stats_func(channel_data, sigma=max(sigma_low, sigma_high), axis=0, maxiters=5) 
            lower_bound = median_ch - sigma_low * stddev_ch; upper_bound = median_ch + sigma_high * stddev_ch
            channel_rejection_this_iter = (channel_data < lower_bound) | (channel_data > upper_bound)
            rejection_mask[..., c] = ~channel_rejection_this_iter
            output_data_with_nans[channel_rejection_this_iter, c] = np.nan
            now = time.time()
            step_times.append(now - last_time)
            last_time = now
            if progress_callback:
                try:
                    progress_callback("stack_kappa", c + 1, total_steps)
                except Exception:
                    pass
    elif stacked_array_NHDWC.ndim == 3: # Monochrome (N, H, W)
        try: _, median_img, stddev_img = sigma_clipped_stats_func(stacked_array_NHDWC, sigma_lower=sigma_low, sigma_upper=sigma_high, axis=0, maxiters=5)
        except TypeError: _, median_img, stddev_img = sigma_clipped_stats_func(stacked_array_NHDWC, sigma=max(sigma_low, sigma_high), axis=0, maxiters=5)
        lower_bound = median_img - sigma_low * stddev_img; upper_bound = median_img + sigma_high * stddev_img
        stats_rejection_this_iter = (stacked_array_NHDWC < lower_bound) | (stacked_array_NHDWC > upper_bound)
        rejection_mask = ~stats_rejection_this_iter
        output_data_with_nans[stats_rejection_this_iter] = np.nan
        if progress_callback:
            try:
                progress_callback("stack_kappa", 1, 1)
            except Exception:
                pass
    else:
        _pcb("stackhelper_error_kappa_sigma_unexpected_shape", lvl="ERROR", shape=stacked_array_NHDWC.shape)
        return stacked_array_NHDWC, rejection_mask
    num_rejected = np.sum(~rejection_mask)
    _pcb("stackhelper_info_kappa_sigma_rejected_pixels", lvl="INFO_DETAIL", num_rejected=num_rejected)
    return output_data_with_nans, rejection_mask


def _apply_winsor_single(args):
    """Helper for parallel winsorization.

    Ensures winsorization is applied along the image axis (axis=0) to
    preserve per-pixel statistics. ``scipy.stats.mstats.winsorize`` flattens
    the array when no axis is provided which would incorrectly clip signal
    across the entire stack.
    """
    arr, limits = args
    # NumPy-based vectorized winsorization for efficiency
    return _winsorize_axis0_numpy(arr, limits)


def parallel_rejwinsor(channels, limits, max_workers, progress_callback=None):
    """Apply winsorization in parallel on a list of arrays."""
    args_list = [(ch, limits) for ch in channels]

    if max_workers <= 1 or len(args_list) <= 1:
        results = []
        start_time = time.time()
        last_time = start_time
        for idx, a in enumerate(args_list, start=1):
            results.append(_apply_winsor_single(a))
            if progress_callback:
                now = time.time()
                progress_callback("stack_winsorized", idx, len(args_list))
                last_time = now
        return results

    results = [None] * len(args_list)

    # Avoid ProcessPool on Windows to prevent heavy memory duplication and startup overhead
    parent_is_daemon = multiprocessing.current_process().daemon
    use_threads = parent_is_daemon or (os.name == 'nt')
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    # Clamp workers to number of channels
    eff_workers = max(1, min(int(max_workers or 1), len(args_list)))

    with Executor(max_workers=eff_workers) as exe:
        futures = {exe.submit(_apply_winsor_single, a): i for i, a in enumerate(args_list)}
        total = len(futures)
        done = 0
        start_time = time.time()
        last_time = start_time
        step_times = []
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
            done += 1
            if progress_callback:
                now = time.time()
                step_times.append(now - last_time)
                last_time = now
                progress_callback("stack_winsorized", done, total)

    return results



def _reject_outliers_winsorized_sigma_clip(
    stacked_array_NHDWC: np.ndarray,
    winsor_limits_tuple: tuple[float, float],  # (low_cut_fraction, high_cut_fraction)
    sigma_low: float,
    sigma_high: float,
    progress_callback: callable = None,
    max_workers: int = 1,
    apply_rewinsor: bool = True,
    weights_chunk: Optional[np.ndarray] = None,
    streaming_state: Optional[WinsorStreamingState] = None,
) -> tuple[np.ndarray | WinsorStreamingState, Optional[np.ndarray]]:
    """
    Rejette les outliers en utilisant un Winsorized Sigma Clip.
    1. Winsorize les données le long de l'axe des images.
    2. Calcule les statistiques sigma-clippées sur les données winsorisées.
    3. Rejette les pixels des données *originales* basés sur ces statistiques.

    Args:
        stacked_array_NHDWC: Tableau des images empilées (N, H, W, C) ou (N, H, W).
        winsor_limits_tuple: Tuple de fractions (0-0.5) pour écrêter en bas et en haut.
        sigma_low: Nombre de sigmas pour le seuil inférieur de rejet.
        sigma_high: Nombre de sigmas pour le seuil supérieur de rejet.
        progress_callback: Fonction de callback pour les logs.
        max_workers: Nombre maximum de travailleurs parallèles pour la winsorisation.
            Typiquement issu de ``run_cfg.winsor_worker_limit``.
        apply_rewinsor: Si True, remplace les pixels rejetés par les valeurs winsorisées
            ("rewinsorisation"). Sinon, les pixels rejetés restent NaN.
        weights_chunk: Poids optionnels pour les images du bloc courant (alignés sur l'axe 0).
        streaming_state: Accumulateur de streaming. Si fourni, la fonction agrège les
            statistiques dans cet objet et retourne ``streaming_state`` en première valeur
            au lieu d'un tableau complet.

    Returns:
        tuple[np.ndarray | WinsorStreamingState, Optional[np.ndarray]]:
            - output_data_with_nans ou l'accumulateur de streaming si ``streaming_state`` est fourni.
            - rejection_mask: Masque booléen (True où les pixels sont gardés) ou ``None`` si le masque complet
              n'est pas construit pour réduire l'empreinte mémoire.
    """
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    if not (SCIPY_AVAILABLE and winsorize_func and SIGMA_CLIP_AVAILABLE and sigma_clipped_stats_func):
        missing_deps = []
        if not SCIPY_AVAILABLE or not winsorize_func: missing_deps.append("Scipy (winsorize)")
        if not SIGMA_CLIP_AVAILABLE or not sigma_clipped_stats_func: missing_deps.append("Astropy.stats (sigma_clipped_stats)")
        _pcb("reject_winsor_warn_deps_unavailable", lvl="WARN", missing=", ".join(missing_deps))
        # Retourner les données originales sans rejet si les dépendances manquent
        data_view = stacked_array_NHDWC.astype(np.float32, copy=False)
        fallback_sum_chunk = np.nansum(data_view, axis=0, dtype=np.float64)
        fallback_count_chunk = np.isfinite(data_view).sum(axis=0, dtype=np.int64)
        if streaming_state is not None:
            streaming_state.update(
                processed_chunk=data_view,
                fallback_sum_chunk=fallback_sum_chunk,
                fallback_count_chunk=fallback_count_chunk,
                weights_chunk=weights_chunk,
            )
            return streaming_state, None
        return data_view, None

    _pcb(f"RejWinsor: Début Rejet Winsorized Sigma Clip. Limits={winsor_limits_tuple}, SigmaLow={sigma_low}, SigmaHigh={sigma_high}.", lvl="DEBUG",
         shape=stacked_array_NHDWC.shape)

    # S'assurer que les limites de winsorisation sont valides (doit être déjà fait dans la GUI, mais double check)
    low_cut, high_cut = winsor_limits_tuple
    if not (0.0 <= low_cut < 0.5 and 0.0 <= high_cut < 0.5 and (low_cut + high_cut) < 1.0):
        _pcb("reject_winsor_error_invalid_limits", lvl="ERROR", limits=winsor_limits_tuple)
        return _finalize_result(output_data_with_nans)

    # Copie des données originales pour y insérer les NaN pour les pixels rejetés
    output_data_with_nans = stacked_array_NHDWC.astype(np.float32, copy=False) # Travailler sur float32
    fallback_sum_chunk = np.nansum(output_data_with_nans, axis=0, dtype=np.float64)
    fallback_count_chunk = np.isfinite(output_data_with_nans).sum(axis=0, dtype=np.int64)

    def _finalize_result(result_array: np.ndarray) -> tuple[np.ndarray | WinsorStreamingState, Optional[np.ndarray]]:
        if streaming_state is not None:
            streaming_state.update(
                processed_chunk=result_array,
                fallback_sum_chunk=fallback_sum_chunk,
                fallback_count_chunk=fallback_count_chunk,
                weights_chunk=weights_chunk,
            )
            return streaming_state, None
        return result_array, None
    total_rejected_pixels = 0

    is_color = stacked_array_NHDWC.ndim == 4 and stacked_array_NHDWC.shape[-1] == 3
    num_images_in_stack = stacked_array_NHDWC.shape[0]

    if num_images_in_stack < 3: # Winsorize et sigma-clip ont besoin d'assez de données
        _pcb("reject_winsor_warn_not_enough_images", lvl="WARN", num_images=num_images_in_stack)
        return _finalize_result(output_data_with_nans)

    try:
        if is_color:
            _pcb("RejWinsor: Traitement image couleur (par canal).", lvl="DEBUG_DETAIL")

            orig_channels = [stacked_array_NHDWC[..., idx].astype(np.float32, copy=False)
                             for idx in range(stacked_array_NHDWC.shape[-1])]

            num_images = stacked_array_NHDWC.shape[0]
            height = stacked_array_NHDWC.shape[1]
            width = stacked_array_NHDWC.shape[2]

            for c_idx, original_channel_data_NHW in enumerate(orig_channels):
                rejected_in_channel = 0

                for rows_slice in _iter_row_chunks(height, num_images, width, original_channel_data_NHW.dtype.itemsize):
                    orig_block = original_channel_data_NHW[:, rows_slice, :]
                    wins_block = _winsorize_block_numpy(orig_block, winsor_limits_tuple)

                    try:
                        _, median_winsorized, stddev_winsorized = sigma_clipped_stats_func(
                            wins_block, sigma=3.0, axis=0, maxiters=5
                        )
                    except TypeError:
                        _, median_winsorized, stddev_winsorized = sigma_clipped_stats_func(
                            wins_block, sigma_lower=3.0, sigma_upper=3.0, axis=0, maxiters=5
                        )

                    lower_bound = median_winsorized - (sigma_low * stddev_winsorized)
                    upper_bound = median_winsorized + (sigma_high * stddev_winsorized)

                    pixels_to_reject_block = (
                        orig_block < lower_bound[np.newaxis, ...]
                    ) | (
                        orig_block > upper_bound[np.newaxis, ...]
                    )

                    rejected_in_channel += int(np.sum(pixels_to_reject_block))
                    channel_view = output_data_with_nans[:, rows_slice, :, c_idx]
                    channel_view[pixels_to_reject_block] = np.nan

                    if apply_rewinsor:
                        np.copyto(channel_view, wins_block, where=np.isnan(channel_view))

                total_rejected_pixels += rejected_in_channel
                _pcb(
                    f"    RejWinsor: Canal {c_idx}, {rejected_in_channel} pixels rejetés.",
                    lvl="DEBUG_DETAIL",
                )
                time.sleep(0)

                if progress_callback:
                    try:
                        progress_callback("stack_winsorized", c_idx + 1, len(orig_channels))
                    except Exception:
                        pass

        else: # Image monochrome (N, H, W)
            _pcb("reject_winsor_info_mono_progress", lvl="INFO_DETAIL")
            _pcb("RejWinsor: Traitement image monochrome.", lvl="DEBUG_DETAIL")
            original_data_NHW = stacked_array_NHDWC.astype(np.float32, copy=False)
            num_images = original_data_NHW.shape[0]
            height = original_data_NHW.shape[1]
            width = original_data_NHW.shape[2]

            num_rejected_mono = 0

            for rows_slice in _iter_row_chunks(height, num_images, width, original_data_NHW.dtype.itemsize):
                orig_block = original_data_NHW[:, rows_slice, :]
                wins_block = _winsorize_block_numpy(orig_block, winsor_limits_tuple)

                try:
                    _, median_winsorized, stddev_winsorized = sigma_clipped_stats_func(
                        wins_block, sigma=3.0, axis=0, maxiters=5
                    )
                except TypeError:
                     _, median_winsorized, stddev_winsorized = sigma_clipped_stats_func(
                        wins_block, sigma_lower=3.0, sigma_upper=3.0, axis=0, maxiters=5
                    )

                lower_bound = median_winsorized - (sigma_low * stddev_winsorized)
                upper_bound = median_winsorized + (sigma_high * stddev_winsorized)

                pixels_to_reject_block = (
                    orig_block < lower_bound[np.newaxis, ...]
                ) | (
                    orig_block > upper_bound[np.newaxis, ...]
                )

                num_rejected_mono += int(np.sum(pixels_to_reject_block))
                output_block = output_data_with_nans[:, rows_slice, :]
                output_block[pixels_to_reject_block] = np.nan

                if apply_rewinsor:
                    np.copyto(output_block, wins_block, where=np.isnan(output_block))

            total_rejected_pixels += num_rejected_mono
            _pcb(f"  RejWinsor: Monochrome, {num_rejected_mono} pixels rejetés.", lvl="DEBUG_DETAIL")

    except MemoryError as e_mem:
        _pcb("reject_winsor_error_memory", lvl="ERROR", error=str(e_mem))
        _internal_logger.error("MemoryError dans _reject_outliers_winsorized_sigma_clip", exc_info=True)
        # En cas de MemoryError, retourner une vue float32 sans duplication pour éviter un double échec.
        fallback_view = stacked_array_NHDWC.astype(np.float32, copy=False)
        return _finalize_result(fallback_view)
    except Exception as e_winsor:
        _pcb("reject_winsor_error_unexpected", lvl="ERROR", error=str(e_winsor))
        _internal_logger.error("Erreur inattendue dans _reject_outliers_winsorized_sigma_clip", exc_info=True)
        return _finalize_result(stacked_array_NHDWC.astype(np.float32, copy=True))

    _pcb("reject_winsor_info_finished", lvl="INFO_DETAIL", num_rejected=total_rejected_pixels)

    output_data_final = output_data_with_nans
    if apply_rewinsor:
        output_data_final = output_data_with_nans

    return _finalize_result(output_data_final)

def _reject_outliers_linear_fit_clip(
    stacked_array_NHDWC: np.ndarray,
    # Quels paramètres seraient nécessaires ?
    # Probablement des seuils pour le rejet des résidus,
    # peut-être des options pour le type de modèle à ajuster.
    # Pour l'instant, gardons-le simple.
    # sigma_clip_low_resid: float = 3.0, # Exemple de paramètre
    # sigma_clip_high_resid: float = 3.0, # Exemple de paramètre
    progress_callback: callable = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rejette les outliers en utilisant un Linear Fit Clipping (PLACEHOLDER).
    Cette méthode vise à modéliser et à soustraire les variations lentes (gradients)
    entre les images et l'image de référence (ex: médiane du stack), puis à rejeter
    les pixels qui s'écartent significativement de ce modèle.

    Args:
        stacked_array_NHDWC: Tableau des images empilées (N, H, W, C) ou (N, H, W).
        progress_callback: Fonction de callback pour les logs.
        // Ajouter d'autres paramètres au besoin.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - output_data_with_nans ou réwinsorisé selon ``apply_rewinsor``.
            - rejection_mask: Masque booléen (True où les pixels sont gardés).
    """
    _pcb = lambda msg_key, lvl="INFO_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else _internal_logger.debug(f"PCB_FALLBACK_{lvl}: {msg_key} {kwargs}")

    _pcb("reject_linearfit_warn_not_implemented", lvl="WARN",
         shape=stacked_array_NHDWC.shape)

    # Pour l'instant, cette fonction ne fait rien et retourne les données telles quelles.
    # L'implémentation réelle serait complexe.
    return stacked_array_NHDWC.copy(), np.ones_like(stacked_array_NHDWC, dtype=bool)


# zemosaic_align_stack.py

# ... (imports et autres fonctions restent les mêmes) ...

def stack_aligned_images(
    aligned_image_data_list: list[np.ndarray | None],
    normalize_method: str = 'none',
    weighting_method: str = 'none',
    rejection_algorithm: str = 'kappa_sigma',
    final_combine_method: str = 'mean',
    sigma_clip_low: float = 3.0,
    sigma_clip_high: float = 3.0,
    winsor_limits: tuple[float, float] = (0.05, 0.05),
    minimum_signal_adu_target: float = 0.0,
    apply_radial_weight: bool = False,
    radial_feather_fraction: float = 0.8,
    radial_shape_power: float = 2.0,
    winsor_max_workers: int = 1,
    progress_callback: callable = None,
    zconfig=None,
    stack_metadata: dict | None = None,
) -> np.ndarray | None:
    """
    Stacke une liste d'images alignées, appliquant normalisation, pondération (qualité + radiale),
    et rejet d'outliers optionnels. VERSION AVEC LOGS DE DEBUG INTENSIFS.
    ``winsor_max_workers`` permet de paralléliser la phase de Winsorisation lors
    du rejet Winsorized Sigma Clip.
    """
    # Wrapper: demote very verbose internal logs so they don't flood the GUI
    def _pcb(msg_key, prog=None, lvl="INFO_DETAIL", **kwargs):
        level = lvl
        try:
            if isinstance(msg_key, str) and msg_key.startswith("STACK_IMG"):
                if isinstance(level, str) and level.upper() in ("ERROR", "INFO"):
                    level = "DEBUG_DETAIL"
        except Exception:
            pass
        if progress_callback:
            return progress_callback(msg_key, prog, level, **kwargs)
        else:
            return _internal_logger.debug(f"PCB_FALLBACK_{level}_{prog}: {msg_key} {kwargs}")

    _pcb("STACK_IMG_ENTRY: Début stack_aligned_images.", lvl="ERROR") # Log d'entrée

    valid_images_to_stack = [img for img in aligned_image_data_list if img is not None and isinstance(img, np.ndarray)]
    if not valid_images_to_stack:
        _pcb("stackimages_warn_no_valid_images", lvl="WARN")
        _pcb("STACK_IMG_EXIT: Retourne None (pas d'images valides).", lvl="ERROR")
        return None

    num_images = len(valid_images_to_stack)
    requested_weight_method = str(weighting_method or "none")
    weighting_method = requested_weight_method.lower().strip() or "none"
    _pcb("stackimages_info_start_stacking", lvl="INFO",
              num_images=num_images, norm=normalize_method,
              weight=weighting_method, reject=rejection_algorithm,
              combine=final_combine_method,
              radial_weight_active=apply_radial_weight,
              radial_feather=radial_feather_fraction if apply_radial_weight else "N/A")

    # --- Préparation des images ---
    first_shape = None
    processed_images_for_stack = []
    for idx, img_adu in enumerate(valid_images_to_stack):
        current_img = img_adu 
        if current_img.dtype != np.float32:
            _pcb(f"StackImages: AVERT Image {idx} pas en float32 ({current_img.dtype}), conversion.", lvl="WARN")
            current_img = current_img.astype(np.float32, copy=False)
        if not current_img.flags.c_contiguous:
            current_img = np.ascontiguousarray(current_img, dtype=np.float32)
        
        # Vérification des infinités DÈS LE DÉBUT
        if not np.all(np.isfinite(current_img)):
            _pcb(f"STACK_IMG_PREP: AVERT Image {idx} (shape {current_img.shape}) contient des non-finis AVANT normalisation. Remplacement par 0.", lvl="ERROR")
            current_img = np.nan_to_num(current_img, nan=0.0, posinf=0.0, neginf=0.0)

        if first_shape is None: first_shape = current_img.shape
        elif current_img.shape != first_shape:
            _pcb("stackimages_warn_inconsistent_shape", lvl="WARN", img_index=idx, shape=current_img.shape, ref_shape=first_shape)
            continue 
        processed_images_for_stack.append(current_img)

    if not processed_images_for_stack:
        _pcb("stackimages_error_no_images_after_shape_check", lvl="ERROR")
        _pcb("STACK_IMG_EXIT: Retourne None (pas d'images après check shape).", lvl="ERROR")
        return None
    
    current_images_data_list = processed_images_for_stack # Renommage pour la suite
    _pcb(f"STACK_IMG_PREP: {len(current_images_data_list)} images prêtes pour normalisation.", lvl="ERROR")


    # --- NORMALISATION ---
    use_gpu_norm = False
    try:
        if zconfig is not None:
            # Normalization during stacking should follow stacking GPU flags only.
            use_gpu_norm = bool(getattr(zconfig, 'stack_use_gpu',
                                        getattr(zconfig, 'use_gpu_stack',
                                                getattr(zconfig, 'use_gpu', False))))
    except Exception:
        use_gpu_norm = False
    if normalize_method == 'linear_fit':
        _pcb("STACK_IMG_NORM: Appel _normalize_images_linear_fit.", lvl="ERROR")
        current_images_data_list = _normalize_images_linear_fit(current_images_data_list, progress_callback=progress_callback, use_gpu=use_gpu_norm) # GPU percentiles if enabled
    elif normalize_method == 'sky_mean':
        _pcb("STACK_IMG_NORM: Appel _normalize_images_sky_mean.", lvl="ERROR")
        current_images_data_list = _normalize_images_sky_mean(current_images_data_list, progress_callback=progress_callback, use_gpu=use_gpu_norm)
    # ... (autres méthodes de normalisation si ajoutées) ...
    
    current_images_data_list = [img for img in current_images_data_list if img is not None] # Filtrer si normalisation a échoué pour certaines
    if not current_images_data_list:
        _pcb("stackimages_error_no_images_left_after_normalization_step", lvl="ERROR")
        _pcb("STACK_IMG_EXIT: Retourne None (pas d'images après normalisation).", lvl="ERROR")
        return None

    _pcb(f"STACK_IMG_NORM: {len(current_images_data_list)} images après normalisation. Vérification des non-finis POST-normalisation.", lvl="ERROR")
    temp_list_post_norm = []
    for idx_post_norm, img_post_norm in enumerate(current_images_data_list):
        if img_post_norm is not None:
            if not np.all(np.isfinite(img_post_norm)):
                _pcb(f"STACK_IMG_NORM: AVERT Image post-norm {idx_post_norm} (shape {img_post_norm.shape}) a des non-finis. Remplacement par 0.", lvl="ERROR")
                img_post_norm = np.nan_to_num(img_post_norm, nan=0.0, posinf=0.0, neginf=0.0)
            temp_list_post_norm.append(img_post_norm)
    current_images_data_list = temp_list_post_norm
    del temp_list_post_norm
    if not current_images_data_list: # Double check
        _pcb("STACK_IMG_NORM: Toutes les images sont devenues None après nettoyage post-normalisation.", lvl="ERROR")
        return None


    # --- PONDÉRATION DE QUALITÉ (Bruit/FWHM) ---
    _pcb(f"STACK_IMG_WEIGHT_QUAL: Début calcul poids qualité. Méthode demandée: {weighting_method}", lvl="ERROR")
    quality_weights_list, effective_weight_method, quality_weight_stats = _compute_quality_weights(
        current_images_data_list,
        weighting_method,
        progress_callback=progress_callback,
    )
    if effective_weight_method != weighting_method:
        _pcb(
            f"STACK_IMG_WEIGHT_QUAL: Fallback méthode '{weighting_method}' -> '{effective_weight_method}'.",
            lvl="WARN",
        )
    weighting_method = effective_weight_method
    if quality_weights_list and quality_weight_stats:
        _pcb(
            f"[Stack] Weights computed (method={weighting_method}): min={quality_weight_stats['min']:.3g} max={quality_weight_stats['max']:.3g}",
            lvl="INFO",
        )
    _pcb(
        "STACK_IMG_WEIGHT_QUAL: Fin calcul poids qualité.",
        lvl="ERROR",
    )
    quality_weights_list = quality_weights_list or []


    # --- PONDÉRATION RADIALE ---
    final_radial_weights_list = [None] * len(current_images_data_list)
    _pcb(f"STACK_IMG_WEIGHT_RAD: Début calcul poids radiaux. Apply: {apply_radial_weight}", lvl="ERROR")
    if apply_radial_weight and ZEMOSAIC_UTILS_AVAILABLE_FOR_RADIAL and make_radial_weight_map_func:
        for idx, img_data_HWC in enumerate(current_images_data_list):
            if img_data_HWC is None: continue
            h, w = img_data_HWC.shape[:2]
            try:
                w_radial_2d = make_radial_weight_map_func(h, w, feather_fraction=radial_feather_fraction, shape_power=radial_shape_power)
                if img_data_HWC.ndim == 3:
                    final_radial_weights_list[idx] = np.repeat(w_radial_2d[..., np.newaxis], img_data_HWC.shape[-1], axis=2).astype(np.float32, copy=False)
                elif img_data_HWC.ndim == 2:
                    final_radial_weights_list[idx] = w_radial_2d.astype(np.float32, copy=False)
            except Exception as e_radw_post: # ... log erreur ...
                final_radial_weights_list[idx] = np.ones_like(img_data_HWC, dtype=np.float32)
    _pcb(f"STACK_IMG_WEIGHT_RAD: Fin calcul poids radiaux. final_radial_weights_list is {'None' if final_radial_weights_list is None else 'Exists'}.", lvl="ERROR")
    if final_radial_weights_list and any(w is not None for w in final_radial_weights_list):
        first_valid_r_weight = next((w for w in final_radial_weights_list if w is not None), None)
        if first_valid_r_weight is not None:
            _pcb(f"STACK_IMG_WEIGHT_RAD: Premier final_radial_weight non-None - shape: {first_valid_r_weight.shape}, type: {first_valid_r_weight.dtype}, range: [{np.min(first_valid_r_weight):.3g}-{np.max(first_valid_r_weight):.3g}]", lvl="ERROR")


    # --- COMBINAISON DES POIDS ---
    image_weights_list_combined = [None] * len(current_images_data_list)
    for i in range(len(current_images_data_list)):
        if current_images_data_list[i] is None: continue
        q_w = quality_weights_list[i] if quality_weights_list and i < len(quality_weights_list) and quality_weights_list[i] is not None else None
        r_w = final_radial_weights_list[i] if final_radial_weights_list[i] is not None else None
        if q_w is not None and r_w is not None: image_weights_list_combined[i] = q_w * r_w
        elif q_w is not None: image_weights_list_combined[i] = q_w
        elif r_w is not None: image_weights_list_combined[i] = r_w
        else: image_weights_list_combined[i] = None
    _pcb(f"STACK_IMG_WEIGHT_COMB: Poids combinés. image_weights_list_combined is {'None' if image_weights_list_combined is None else 'Exists'}.", lvl="ERROR")
    if image_weights_list_combined and any(w is not None for w in image_weights_list_combined):
        first_valid_c_weight = next((w for w in image_weights_list_combined if w is not None), None)
        if first_valid_c_weight is not None:
             _pcb(f"STACK_IMG_WEIGHT_COMB: Premier poids combiné non-None - shape: {first_valid_c_weight.shape}, type: {first_valid_c_weight.dtype}, range: [{np.min(first_valid_c_weight):.3g}-{np.max(first_valid_c_weight):.3g}]", lvl="ERROR")


    # --- STACKAGE NUMPY ---
    try:
        # S'assurer que current_images_data_list ne contient que des images valides avant stack
        valid_images_for_numpy_stack = [img for img in current_images_data_list if img is not None]
        if not valid_images_for_numpy_stack:
            _pcb("stackimages_error_no_images_to_stack_before_np_stack", lvl="ERROR")
            _pcb("STACK_IMG_EXIT: Retourne None (pas d'images avant np.stack).", lvl="ERROR")
            return None
            
        stacked_array_NHDWC = np.stack(valid_images_for_numpy_stack, axis=0)
        _pcb(f"STACK_IMG_NP_STACK: stacked_array_NHDWC - shape: {stacked_array_NHDWC.shape}, dtype: {stacked_array_NHDWC.dtype}, range: [{np.min(stacked_array_NHDWC):.2g}-{np.max(stacked_array_NHDWC):.2g}]", lvl="ERROR")

        # Filtrer les poids combinés pour correspondre EXACTEMENT aux images stackées
        filtered_combined_weights = [
            image_weights_list_combined[i] 
            for i, img in enumerate(current_images_data_list) # Itérer sur la liste originale avant filtrage pour garder les bons indices de poids
            if img is not None # Condition pour que l'image soit dans valid_images_for_numpy_stack
        ]
        del current_images_data_list, valid_images_for_numpy_stack # Libérer mémoire
        gc.collect()
    except Exception as e_np_stack:
        _pcb(f"stackimages_error_value_stacking_images: {e_np_stack}", lvl="ERROR")
        _pcb("STACK_IMG_EXIT: Retourne None (erreur np.stack).", lvl="ERROR")
        return None

    weights_array_NHDWC = None
    if filtered_combined_weights and any(w is not None for w in filtered_combined_weights):
        try:
            # Prendre uniquement les poids qui ne sont pas None
            valid_weights_to_stack_numpy = [w for w in filtered_combined_weights if w is not None]
            if not valid_weights_to_stack_numpy:
                _pcb("STACK_IMG_NP_STACK_WEIGHTS: Tous les poids filtrés sont None. weights_array_NHDWC sera None.", lvl="ERROR")
            elif len(valid_weights_to_stack_numpy) != stacked_array_NHDWC.shape[0]:
                 _pcb(f"STACK_IMG_NP_STACK_WEIGHTS: ERREUR - Mismatch nombre poids valides ({len(valid_weights_to_stack_numpy)}) et images stackées ({stacked_array_NHDWC.shape[0]}).", lvl="ERROR")
            else:
                weights_array_NHDWC = np.stack(valid_weights_to_stack_numpy, axis=0)
                if weights_array_NHDWC.shape != stacked_array_NHDWC.shape:
                    _pcb(f"stackimages_warn_combined_weights_shape_mismatch_final. Shape poids: {weights_array_NHDWC.shape}, Shape data: {stacked_array_NHDWC.shape}", lvl="ERROR")
                    weights_array_NHDWC = None 
        except Exception as e_w_stack:
            _pcb(f"stackimages_error_stacking_combined_weights: {e_w_stack}", lvl="ERROR")
            weights_array_NHDWC = None # S'assurer qu'il est None en cas d'erreur
    
    _pcb(f"STACK_IMG_NP_STACK_WEIGHTS: weights_array_NHDWC is {'None' if weights_array_NHDWC is None else 'Exists'}.", lvl="ERROR")
    if weights_array_NHDWC is not None:
         _pcb(f"STACK_IMG_NP_STACK_WEIGHTS: weights_array_NHDWC - shape: {weights_array_NHDWC.shape}, type: {weights_array_NHDWC.dtype}, range: [{np.min(weights_array_NHDWC):.3g}-{np.max(weights_array_NHDWC):.3g}]", lvl="ERROR")

    # ... (Nettoyage des listes de poids intermédiaires) ...
    del quality_weights_list, final_radial_weights_list, image_weights_list_combined, filtered_combined_weights
    gc.collect()

    # --- REJET D'OUTLIERS ---
    _pcb(f"STACK_IMG_REJECT: Début rejet. Algorithme: {rejection_algorithm}", lvl="ERROR")
    data_for_combine = stacked_array_NHDWC 
    rejection_mask = None
    if rejection_algorithm == 'kappa_sigma':
        data_for_combine, rejection_mask = _reject_outliers_kappa_sigma(stacked_array_NHDWC, sigma_clip_low, sigma_clip_high, progress_callback)
    elif rejection_algorithm == 'winsorized_sigma_clip':
        data_for_combine, rejection_mask = _reject_outliers_winsorized_sigma_clip(
            stacked_array_NHDWC,
            winsor_limits,
            sigma_clip_low,
            sigma_clip_high,
            progress_callback,
            winsor_max_workers,
        )
    # ... (autres algos de rejet) ...
    _pcb(f"STACK_IMG_REJECT: Fin rejet. data_for_combine shape: {data_for_combine.shape}, range: [{np.nanmin(data_for_combine):.2g}-{np.nanmax(data_for_combine):.2g}] (contient NaN)", lvl="ERROR")


    # --- COMBINAISON FINALE ---
    _pcb(f"STACK_IMG_COMBINE: Début combinaison finale. Méthode: {final_combine_method}", lvl="ERROR")
    result_image_adu = None
    try:
        effective_weights_for_combine = weights_array_NHDWC
        if effective_weights_for_combine is not None:
            if rejection_mask is None:
                mask_keep = np.isfinite(data_for_combine)
            else:
                mask_keep = rejection_mask
            _pcb(f"STACK_IMG_COMBINE: Application masque de rejet aux poids.", lvl="ERROR")
            effective_weights_for_combine = np.where(mask_keep, weights_array_NHDWC, 0.0)
            if rejection_mask is None:
                del mask_keep
        
        _pcb(f"STACK_IMG_COMBINE: effective_weights_for_combine is {'None' if effective_weights_for_combine is None else 'Exists'}.", lvl="ERROR")
        if effective_weights_for_combine is not None:
            _pcb(f"STACK_IMG_COMBINE: effective_weights_for_combine - shape: {effective_weights_for_combine.shape}, dtype: {effective_weights_for_combine.dtype}, range: [{np.min(effective_weights_for_combine):.3g}-{np.max(effective_weights_for_combine):.3g}]", lvl="ERROR")


        if final_combine_method == 'mean':
            if effective_weights_for_combine is None:
                _pcb("STACK_IMG_COMBINE_MEAN: Pas de poids effectifs, utilisation np.nanmean.", lvl="ERROR")
                result_image_adu = np.nanmean(data_for_combine, axis=0)
            else:
                if data_for_combine.shape[0] != effective_weights_for_combine.shape[0]:
                    _pcb("stackimages_error_mean_combine_shape_mismatch (data vs effective_weights)", lvl="ERROR", 
                         data_N=data_for_combine.shape[0], weights_N=effective_weights_for_combine.shape[0])
                    result_image_adu = np.nanmean(data_for_combine, axis=0)
                else:
                    data_masked_for_avg = np.nan_to_num(data_for_combine, nan=0.0)
                    _pcb(f"STACK_IMG_COMBINE_MEAN: data_masked_for_avg - shape: {data_masked_for_avg.shape}, dtype: {data_masked_for_avg.dtype}, range: [{np.min(data_masked_for_avg):.2g}-{np.max(data_masked_for_avg):.2g}]", lvl="ERROR")
                    
                    if data_masked_for_avg.shape[0] == 1: # Cas N=1
                        _pcb(f"STACK_IMG_COMBINE_MEAN: N=1. Multiplication directe: data * poids_effectif.", lvl="ERROR")
                        # Log des pixels avant et après
                        img_idx_log, h_log, w_log, c_log = 0,0,0,0 # Pixel (0,0,0) de la première (unique) image
                        _pcb(f"  N=1 PRE-MULT: data_pixel=[{data_masked_for_avg[img_idx_log,h_log,w_log,c_log]:.3g}], weight_pixel=[{effective_weights_for_combine[img_idx_log,h_log,w_log,c_log]:.3g}]", lvl="ERROR")
                        result_image_adu = data_masked_for_avg[0] * effective_weights_for_combine[0]
                        _pcb(f"  N=1 POST-MULT: result_pixel=[{result_image_adu[h_log,w_log,c_log]:.3g}]", lvl="ERROR")
                    else: # Cas N > 1
                        _pcb(f"STACK_IMG_COMBINE_MEAN: N={data_masked_for_avg.shape[0]}. Moyenne pondérée standard.", lvl="ERROR")
                        sum_weighted_data = np.sum(data_masked_for_avg * effective_weights_for_combine, axis=0)
                        sum_weights = np.sum(effective_weights_for_combine, axis=0)
                        _pcb(f"  N>1 POST-SUM: sum_weighted_data range: [{np.min(sum_weighted_data):.2g}-{np.max(sum_weighted_data):.2g}]", lvl="ERROR")
                        _pcb(f"  N>1 POST-SUM: sum_weights range: [{np.min(sum_weights):.2g}-{np.max(sum_weights):.2g}]", lvl="ERROR")
                        
                        # Inspecter un pixel de bord pour sum_weights
                        bh,bw = 0,0 # Bord supérieur gauche
                        if sum_weights.ndim == 3 and sum_weights.shape[0]>bh and sum_weights.shape[1]>bw:
                             _pcb(f"  N>1 POST-SUM: sum_weights[{bh},{bw},0] = {sum_weights[bh,bw,0]:.3g}", lvl="ERROR")
                        
                        result_image_adu = np.divide(sum_weighted_data, sum_weights,
                                                   out=np.zeros_like(sum_weighted_data, dtype=np.float32),
                                                   where=sum_weights > 1e-9)
                        _pcb(f"  N>1 POST-DIVIDE: result_image_adu range: [{np.min(result_image_adu):.2g}-{np.max(result_image_adu):.2g}]", lvl="ERROR")
                        if result_image_adu.ndim == 3 and result_image_adu.shape[0]>bh and result_image_adu.shape[1]>bw:
                             _pcb(f"  N>1 POST-DIVIDE: result_image_adu[{bh},{bw},0] = {result_image_adu[bh,bw,0]:.3g}", lvl="ERROR")


        elif final_combine_method == 'median':
            # ... (logique pour median) ...
            if effective_weights_for_combine is not None:
                _pcb("stackimages_warn_median_with_weights_not_supported_simple", lvl="WARN")
            result_image_adu = np.nanmedian(data_for_combine, axis=0)
        # ... (else pour unknown combine method) ...

        if result_image_adu is not None and not np.all(np.isfinite(result_image_adu)):
            _pcb("STACK_IMG_COMBINE: AVERT - result_image_adu contient des non-finis POST-combinaison. Remplacement par 0.", lvl="ERROR")
            result_image_adu = np.nan_to_num(result_image_adu, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception as e_comb_final:
        _pcb(f"stackimages_error_exception_final_combine: {e_comb_final}", lvl="ERROR")
        _internal_logger.error("Erreur combinaison finale dans stack_aligned_images", exc_info=True)
        _pcb("STACK_IMG_EXIT: Retourne None (erreur combinaison finale).", lvl="ERROR")
        return None
    finally:
        # ... (Nettoyage des gros tableaux) ...
        del data_for_combine, rejection_mask, stacked_array_NHDWC
        if weights_array_NHDWC is not None: del weights_array_NHDWC
        if 'effective_weights_for_combine' in locals() and effective_weights_for_combine is not None: del effective_weights_for_combine
        gc.collect()

    if result_image_adu is None:
        _pcb("stackimages_error_combine_result_none", lvl="ERROR")
        _pcb("STACK_IMG_EXIT: Retourne None (result_image_adu est None après try-except).", lvl="ERROR")
        return None

    _pcb(f"STACK_IMG_OFFSET: Avant offset final. Range: [{np.nanmin(result_image_adu):.2g}-{np.nanmax(result_image_adu):.2g}]", lvl="ERROR")
    # ... (Application de minimum_signal_adu_target) ...
    if result_image_adu is not None and minimum_signal_adu_target > 0.0:
        current_min_val = np.nanmin(result_image_adu)
        if np.isfinite(current_min_val) and current_min_val < minimum_signal_adu_target:
            offset_to_apply = minimum_signal_adu_target - current_min_val
            result_image_adu += offset_to_apply
    
    _pcb(f"STACK_IMG_EXIT: Fin stack_aligned_images. Range final: [{np.nanmin(result_image_adu):.2g}-{np.nanmax(result_image_adu):.2g}]", lvl="ERROR")
    if result_image_adu is None:
        return None
    result_image_adu = result_image_adu.astype(np.float32, copy=False)
    _poststack_rgb_equalization(result_image_adu, zconfig, stack_metadata)
    return result_image_adu






# --- CPU fallback implementation appended (in case Seestar stack methods are unavailable) ---
def _cpu_stack_kappa_fallback(
    frames,
    *,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    weights=None,
    progress_callback=None,
):
    """CPU kappa-sigma clip fallback.

    Accepts frames of shape (N,H,W) or (N,H,W,C). Returns (stacked, rejected_pct).
    """
    _pcb = lambda m, lvl="DEBUG_DETAIL", **kw: (
        progress_callback(m, None, lvl, **kw) if progress_callback else _internal_logger.debug(m)
    )
    frames_list = [np.asarray(f, dtype=np.float32) for f in frames]
    if not frames_list:
        raise ValueError("frames is empty")
    arr = np.stack(frames_list, axis=0)
    if arr.ndim not in (3, 4):
        raise ValueError(f"frames must be (N,H,W) or (N,H,W,C); got {arr.shape}")
    med = np.nanmedian(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    low = med - float(sigma_low) * std
    high = med + float(sigma_high) * std
    mask = (arr >= low) & (arr <= high)
    arr_clip = np.where(mask, arr, np.nan)
    if weights is None:
        stacked = np.nanmean(arr_clip, axis=0).astype(np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim == 1 and w.shape[0] == arr.shape[0]:
            extra_dims = (1,) * (arr.ndim - 1)
            w = w.reshape((arr.shape[0],) + extra_dims)
        try:
            bw = np.broadcast_to(w, arr_clip.shape).astype(np.float32, copy=False)
        except ValueError as exc:
            raise ValueError(
                f"weights shape {w.shape} not compatible with frames shape {arr.shape}"
            ) from exc
        bw = np.where(mask, bw, 0.0)
        num = np.nansum(arr_clip * bw, axis=0)
        den = np.sum(bw, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            stacked = np.where(den > 0, num / den, np.nan)
        if np.any(~np.isfinite(stacked)):
            stacked_fallback = np.nanmean(arr, axis=0)
            stacked = np.where(np.isfinite(stacked), stacked, stacked_fallback)
        stacked = stacked.astype(np.float32, copy=False)
    rejected_pct = 100.0 * float(mask.size - np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
    _pcb("cpu_kappa_fallback_done", lvl="DEBUG_DETAIL")
    return stacked, rejected_pct


def _cpu_stack_linear_fallback(
    frames,
    *,
    sigma: float = 3.0,
    weights=None,
    progress_callback=None,
):
    """CPU linear residual clipping fallback.

    Compute per-pixel residuals to the median, clip within sigma*std of residuals,
    and mean-combine. Returns (stacked, rejected_pct).
    """
    _pcb = lambda m, lvl="DEBUG_DETAIL", **kw: (
        progress_callback(m, None, lvl, **kw) if progress_callback else _internal_logger.debug(m)
    )
    frames_list = [np.asarray(f, dtype=np.float32) for f in frames]
    if not frames_list:
        raise ValueError("frames is empty")
    arr = np.stack(frames_list, axis=0)
    if arr.ndim not in (3, 4):
        raise ValueError(f"frames must be (N,H,W) or (N,H,W,C); got {arr.shape}")
    med = np.nanmedian(arr, axis=0)
    resid = arr - med
    med_r = np.nanmedian(resid, axis=0)
    std_r = np.nanstd(resid, axis=0)
    mask = np.abs(resid - med_r) <= float(sigma) * std_r
    arr_clip = np.where(mask, arr, np.nan)
    if weights is None:
        stacked = np.nanmean(arr_clip, axis=0).astype(np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim == 1 and w.shape[0] == arr.shape[0]:
            extra_dims = (1,) * (arr.ndim - 1)
            w = w.reshape((arr.shape[0],) + extra_dims)
        try:
            bw = np.broadcast_to(w, arr_clip.shape).astype(np.float32, copy=False)
        except ValueError as exc:
            raise ValueError(
                f"weights shape {w.shape} not compatible with frames shape {arr.shape}"
            ) from exc
        bw = np.where(mask, bw, 0.0)
        num = np.nansum(arr_clip * bw, axis=0)
        den = np.sum(bw, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            stacked = np.where(den > 0, num / den, np.nan)
        if np.any(~np.isfinite(stacked)):
            stacked_fallback = np.nanmean(arr, axis=0)
            stacked = np.where(np.isfinite(stacked), stacked, stacked_fallback)
        stacked = stacked.astype(np.float32, copy=False)
    rejected_pct = 100.0 * float(mask.size - np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
    _pcb("cpu_linear_fallback_done", lvl="DEBUG_DETAIL")
    return stacked, rejected_pct
def _cpu_stack_winsorized_fallback(
    frames,
    *,
    kappa: float = 3.0,
    winsor_limits: tuple[float, float] = (0.05, 0.05),
    apply_rewinsor: bool = True,
    weights=None,
    winsor_max_workers: int = 1,
    progress_callback=None,
):
    """
    CPU fallback for winsorized sigma-clip stacking.

    Parameters
    ----------
    frames : array-like
        Stack of frames (N,H,W) or (N,H,W,C).
    kappa : float
        Sigma threshold used for clipping (both lower/upper).
    winsor_limits : (float, float)
        Fractions for lower and upper winsorization on axis 0.
    apply_rewinsor : bool
        If True, replace rejected pixels with winsorized values before averaging.
    weights : array-like or None
        Optional per-frame weights. Accepts shapes (N,), (N,1,1[,1]) or (N,H,W[,C]).
    winsor_max_workers : int
        Parallelism hint for winsor steps (passed through to helper).
    progress_callback : callable or None
        Optional progress logger.

    Returns
    -------
    (np.ndarray, float)
        Stacked image (H,W[,C]) as float32 and rejected percentage.
    """
    # Normalize and validate input frames without forcing a ragged ndarray
    frames_list: list[np.ndarray] = []
    first_shape = None
    dropped = 0
    total_seen = 0
    for f in frames:
        total_seen += 1
        a = np.asarray(f, dtype=np.float32)
        if first_shape is None:
            first_shape = a.shape
        if a.shape != first_shape:
            dropped += 1
            if progress_callback:
                try:
                    progress_callback("stack_winsorized_warn_mismatched_shape_dropped", dropped, 0)
                except Exception:
                    pass
            else:
                _internal_logger.warning(
                    f"Winsorized fallback: frame dropped due to shape mismatch."
                    f" expected {first_shape}, got {a.shape}"
                )
            continue
        frames_list.append(a)

    if not frames_list:
        raise ValueError("No frames with consistent shape to stack in CPU winsorized fallback")

    if dropped:
        total_frames = total_seen if total_seen else (dropped + len(frames_list))
        _internal_logger.warning(
            "Winsorized fallback: using %d frames; dropped %d mismatched (total input %d).",
            len(frames_list),
            dropped,
            total_frames,
        )

    arr_shape = (len(frames_list),) + first_shape  # type: ignore[operator]
    dtype = np.dtype(np.float32)
    estimated_bytes = int(np.prod(arr_shape, dtype=np.int64)) * dtype.itemsize
    memmap_path: Optional[str] = None
    if estimated_bytes >= CPU_WINSOR_MEMMAP_THRESHOLD_BYTES:
        tmp = tempfile.NamedTemporaryFile(prefix="zemosaic_winsor_", suffix=".dat", delete=False)
        memmap_path = tmp.name
        tmp.close()
        arr = np.memmap(memmap_path, mode="w+", dtype=dtype, shape=arr_shape)
        for idx, frame in enumerate(frames_list):
            arr[idx] = frame
        arr.flush()
        _internal_logger.debug(
            "CPU winsorized fallback: frames stored in memmap %s (~%.2f GiB).",
            memmap_path,
            estimated_bytes / (1024**3),
        )
    else:
        arr = np.stack(frames_list, axis=0)
    # Libérer au plus tôt la mémoire retenue par la liste de frames individuelles.
    frames_list.clear()
    if arr.ndim not in (3, 4):
        raise ValueError(f"frames must be (N,H,W) or (N,H,W,C); got shape {arr.shape}")

    result: np.ndarray
    rejected_pct_value: float
    try:
        # Compute reject mask and rewinsorized data using existing helper
        data_with_nans, keep_mask = _reject_outliers_winsorized_sigma_clip(
            stacked_array_NHDWC=arr,
            winsor_limits_tuple=winsor_limits,
            sigma_low=kappa,
            sigma_high=kappa,
            progress_callback=progress_callback,
            max_workers=winsor_max_workers,
            apply_rewinsor=apply_rewinsor,
        )

        # Compute rejection percentage
        if keep_mask is None:
            total = data_with_nans.size
            kept = np.count_nonzero(np.isfinite(data_with_nans))
        else:
            total = keep_mask.size
            kept = np.count_nonzero(keep_mask)
        rejected_pct = 100.0 * float(total - kept) / float(total) if total else 0.0

        # Weighted or unweighted combine along axis 0
        if weights is None:
            stacked = np.nanmean(data_with_nans, axis=0)
        else:
            w = np.asarray(weights, dtype=np.float32)
            # Normalize weight shape to broadcast over (N,H/W[,C])
            if w.ndim == 1 and w.shape[0] == arr.shape[0]:
                # reshape to (N,1,1[,1])
                extra_dims = (1,) * (arr.ndim - 1)
                w = w.reshape((arr.shape[0],) + extra_dims)
            # If weights do not align to arr shape except for axis 0, attempt broadcast
            try:
                bw = np.broadcast_to(w, data_with_nans.shape).astype(np.float32, copy=False)
            except Exception:
                raise ValueError(
                    f"weights shape {w.shape} not compatible with frames shape {arr.shape}"
                )
            # Zero-out weights where value is NaN to exclude rejected pixels when apply_rewinsor=False
            mask_finite = np.isfinite(data_with_nans)
            bw = np.where(mask_finite, bw, 0.0)
            num = np.nansum(bw * data_with_nans, axis=0)
            den = np.sum(bw, axis=0)
            # Avoid division by zero: fallback to nanmean of original frames where den==0
            with np.errstate(invalid="ignore", divide="ignore"):
                stacked = np.where(den > 0, num / den, np.nan)
            # Fill any remaining NaNs with simple nanmean across original frames
            if np.any(~np.isfinite(stacked)):
                stacked_fallback = np.nanmean(arr, axis=0)
                stacked = np.where(np.isfinite(stacked), stacked, stacked_fallback)

        result = stacked.astype(np.float32)
        rejected_pct_value = float(rejected_pct)
    finally:
        # Drop references before cleaning the backing file to ease GC on Windows.
        try:
            del data_with_nans
        except UnboundLocalError:
            pass
        try:
            del keep_mask
        except UnboundLocalError:
            pass
        _cleanup_memmap(arr, memmap_path)
    return result, rejected_pct_value

# Bind fallback if seestar CPU implementation is unavailable
try:
    cpu_stack_winsorized
except NameError:
    cpu_stack_winsorized = None
if cpu_stack_winsorized is None:
    cpu_stack_winsorized = _cpu_stack_winsorized_fallback

# Bind other fallbacks when external Seestar stack methods are unavailable
try:
    cpu_stack_kappa
except NameError:
    cpu_stack_kappa = None
if cpu_stack_kappa is None:
    cpu_stack_kappa = _cpu_stack_kappa_fallback

try:
    cpu_stack_linear
except NameError:
    cpu_stack_linear = None
if cpu_stack_linear is None:
    cpu_stack_linear = _cpu_stack_linear_fallback
