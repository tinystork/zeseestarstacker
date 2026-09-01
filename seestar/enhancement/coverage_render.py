"""COV-04: final-only coverage-aware cosmetic reconstruction (render).

This is a *render* operation, never part of scientific accumulation.  It
regularizes noise/transition visibility in low-support regions without
inventing signal.
"""

import numpy as np


def coverage_aware_render(
    sci,
    neff_support,
    *,
    n_ref=32.0,
    sigma_denoise=2.0,
    sigma_low=32.0,
):
    """Blend a denoised detail residual in low-support regions.

    SCI = B + D (low-frequency background + high-frequency detail).
    RENDER = B + (1 - alpha) * D + alpha * D_denoised, where
    ``alpha = clip(1 - N_eff_support / n_ref, 0, 1)``.

    Regions with ``N_eff_support >= n_ref`` (alpha ~ 0) are untouched;
    regions with low ``N_eff_support`` (alpha ~ 1) get progressively stronger
    noise regularization of the detail residual only.

    Honest constraints (non-negotiable):
    * no brightness gain for low coverage (flat fields stay flat);
    * no generative inpainting / invented stars or nebulosity;
    * no low-frequency signal modification driven by WHT;
    * only the high-frequency residual is attenuated in low-support areas.

    Parameters
    ----------
    sci : (H, W) or (H, W, C) float array
        Scientific stack (already normalized or raw).
    neff_support : (H, W) float array
        Positive effective-support (N_eff_support = SUP_W1**2 / SUP_W2).
    n_ref : float
        Effective-support level at or above which alpha = 0 (fully reliable).
    sigma_denoise : float
        Gaussian sigma for the denoised detail residual.
    sigma_low : float
        Gaussian sigma separating the low-frequency background from detail.
    """
    sci = np.asarray(sci, dtype=np.float32)
    sup = np.asarray(neff_support, dtype=np.float32)
    if sup.ndim != 2:
        raise ValueError("neff_support must be 2-D")
    if sci.shape[:2] != sup.shape[:2]:
        raise ValueError("sci and neff_support spatial shapes must match")

    n_ref = float(n_ref)
    if not np.isfinite(n_ref) or n_ref <= 0.0:
        raise ValueError("n_ref must be finite > 0")
    sup_max = float(np.nanmax(sup)) if sup.size else 0.0
    if not np.isfinite(sup_max) or sup_max <= 0.0:
        # no support information -> do not regularize
        return sci.astype(np.float32)

    alpha = np.clip(1.0 - sup / n_ref, 0.0, 1.0).astype(np.float32)

    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        gaussian_filter = None

    if gaussian_filter is None:
        return sci.astype(np.float32)

    out = np.empty_like(sci)
    if sci.ndim == 2:
        B = gaussian_filter(sci, sigma=sigma_low)
        D = sci - B
        Dd = gaussian_filter(D, sigma=sigma_denoise)
        out = B + (1.0 - alpha) * D + alpha * Dd
    else:
        C = sci.shape[2]
        for c in range(C):
            ch = sci[..., c]
            B = gaussian_filter(ch, sigma=sigma_low)
            D = ch - B
            Dd = gaussian_filter(D, sigma=sigma_denoise)
            out[..., c] = B + (1.0 - alpha) * D + alpha * Dd
    return out.astype(np.float32)
