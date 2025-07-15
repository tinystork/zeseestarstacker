import numpy as np
from astropy.stats import sigma_clipped_stats, SigmaClip
try:
    from photutils.background import Background2D, MedianBackground
    from photutils.detection import DAOStarFinder
    from photutils.segmentation import detect_sources
    from photutils.segmentation import SourceCatalog
    _PHOTUTILS_AVAILABLE = True
except Exception:
    _PHOTUTILS_AVAILABLE = False

    class Background2D:  # type: ignore
        pass

    class MedianBackground:  # type: ignore
        pass

    class DAOStarFinder:  # type: ignore
        pass

    def detect_sources(*a, **k):  # type: ignore
        return None

    class SourceCatalog:  # type: ignore
        pass


def _calculate_image_weights_noise_variance(image_list, progress_callback=None):
    """Return per-pixel weights inversely proportional to noise variance."""
    weights = []
    if not image_list:
        return weights
    for img in image_list:
        if img is None:
            weights.append(None)
            continue
        data = img.astype(np.float32, copy=False)
        if data.ndim == 3 and data.shape[-1] == 3:
            chans = []
            for c in range(3):
                _, _, std = sigma_clipped_stats(data[..., c], sigma=3.0, maxiters=5)
                var = std ** 2 if np.isfinite(std) and std > 1e-9 else np.inf
                chans.append(var)
            min_var = np.min([v for v in chans if np.isfinite(v)] + [1e-9])
            w = np.stack([(min_var / v if np.isfinite(v) and v > 0 else 1e-6) * np.ones_like(data[..., i], dtype=np.float32) for i, v in enumerate(chans)], axis=-1)
        else:
            _, _, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
            var = std ** 2 if np.isfinite(std) and std > 1e-9 else np.inf
            min_var = var if np.isfinite(var) else 1e-9
            w = (min_var / var if np.isfinite(var) and var > 0 else 1e-6) * np.ones_like(data, dtype=np.float32)

        # Normalise each weight map so its maximum value is 1.0
        w_max = np.nanmax(w)
        if np.isfinite(w_max) and w_max > 0:
            w = w / w_max

        weights.append(w)
    return weights


def _estimate_initial_fwhm(data_2d):
    try:
        _, median, std = sigma_clipped_stats(data_2d, sigma=3.0, maxiters=5)
        threshold = median + 3.0 * std
        segm = detect_sources(data_2d, threshold, npixels=5)
        if segm is None:
            return 4.0
        cat = SourceCatalog(data_2d, segm)
        fwhms = [p.equivalent_fwhm.value for p in cat if p.eccentricity is not None and p.eccentricity < 0.5 and p.equivalent_fwhm is not None]
        if not fwhms:
            return 4.0
        fwhm = np.nanmedian(fwhms)
        return float(fwhm) if np.isfinite(fwhm) else 4.0
    except Exception:
        return 4.0


def _calculate_image_weights_noise_fwhm(image_list, progress_callback=None):
    weights = []
    if not image_list:
        return weights
    for img in image_list:
        if img is None:
            weights.append(None)
            continue
        data = img.astype(np.float32, copy=False)
        if data.ndim == 3 and data.shape[-1] == 3:
            luminance = 0.299 * data[...,0] + 0.587 * data[...,1] + 0.114 * data[...,2]
        else:
            luminance = data
        if luminance.size < 50*50:
            weights.append(np.ones_like(data, dtype=np.float32))
            continue
        fwhm_est = _estimate_initial_fwhm(luminance)
        try:
            sigma_clip = SigmaClip(sigma=3.0)
            bkg = Background2D(luminance, (32,32), filter_size=(3,3), sigma_clip=sigma_clip, bkg_estimator=MedianBackground())
            data_sub = luminance - bkg.background
            threshold = 5.0 * bkg.background_rms
        except Exception:
            data_sub = luminance - np.median(luminance)
            threshold = 5.0 * np.std(luminance)
        try:
            daofind = DAOStarFinder(fwhm=fwhm_est, threshold=threshold)
            sources = daofind(data_sub)
        except Exception:
            sources = None
        if sources is None or len(sources) < 5:
            weights.append(np.ones_like(data, dtype=np.float32))
            continue
        segm = detect_sources(data_sub, 1.5 * threshold, npixels=7)
        if segm is None:
            weights.append(np.ones_like(data, dtype=np.float32))
            continue
        cat = SourceCatalog(data_sub, segm)
        fwhms = [p.equivalent_fwhm.value for p in cat if p.equivalent_fwhm is not None]
        if not fwhms:
            weights.append(np.ones_like(data, dtype=np.float32))
            continue
        median_fwhm = np.nanmedian(fwhms)
        min_fwhm = max(0.5, np.nanmin(fwhms))
        scalar_weight = min_fwhm / median_fwhm if median_fwhm > 0 else 1.0
        w = np.full_like(data, scalar_weight, dtype=np.float32)

        # Normalise weight map so the maximum equals 1.0
        w_max = np.nanmax(w)
        if np.isfinite(w_max) and w_max > 0:
            w = w / w_max

        weights.append(w)
    return weights
