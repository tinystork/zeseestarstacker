import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter


def calculate_fwhm_ecc(data, fwhm_guess=3.5, threshold_sigma=5.):
    """Calculate median FWHM and eccentricity of stars in ``data``.

    Parameters
    ----------
    data : 2-D numpy array
        Image data.
    fwhm_guess : float
        Initial FWHM guess for DAOStarFinder.
    threshold_sigma : float
        Detection threshold in sigma units.

    Returns
    -------
    fwhm_median_px : float
        Median FWHM of detected stars in pixels, or ``np.nan`` if none.
    ecc_median : float
        Median eccentricity of detected stars, or ``np.nan`` if none.
    n_detected : int
        Number of detected stars.
    """
    try:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        finder = DAOStarFinder(fwhm=fwhm_guess, threshold=threshold_sigma * std)
        tbl = finder(data - median)
        if tbl is None or len(tbl) == 0:
            return np.nan, np.nan, 0

        fwhm_list = []
        ecc_list = []
        size = int(max(3, round(fwhm_guess * 3)))

        for star in tbl:
            x = int(round(star['xcentroid']))
            y = int(round(star['ycentroid']))
            x1 = max(x - size, 0)
            x2 = min(x + size + 1, data.shape[1])
            y1 = max(y - size, 0)
            y2 = min(y + size + 1, data.shape[0])
            cutout = data[y1:y2, x1:x2]
            if cutout.size == 0:
                continue

            cutout = cutout - np.median(cutout)
            cutout = gaussian_filter(cutout, sigma=0.5)
            peak = cutout.max()
            if peak <= 0:
                continue
            half_max = peak / 2.0
            y_profile = cutout.max(axis=1)
            x_profile = cutout.max(axis=0)

            def width_at_half(profile):
                idx = np.where(profile >= half_max)[0]
                if idx.size == 0:
                    return np.nan
                return idx[-1] - idx[0]

            fwhm_x = width_at_half(x_profile)
            fwhm_y = width_at_half(y_profile)
            if not (np.isfinite(fwhm_x) and np.isfinite(fwhm_y)):
                continue

            fwhm_avg = 0.5 * (fwhm_x + fwhm_y)
            fwhm_list.append(fwhm_avg)

            a = max(fwhm_x, fwhm_y) / 2.0
            b = min(fwhm_x, fwhm_y) / 2.0
            ecc = np.sqrt(1.0 - (b / a) ** 2) if a > 0 else np.nan
            ecc_list.append(ecc)

        if not fwhm_list:
            return np.nan, np.nan, 0

        fwhm_med = np.nanmedian(fwhm_list)
        ecc_med = np.nanmedian(ecc_list) if ecc_list else np.nan
        return float(fwhm_med), float(ecc_med), len(fwhm_list)
    except Exception:
        return np.nan, np.nan, 0
