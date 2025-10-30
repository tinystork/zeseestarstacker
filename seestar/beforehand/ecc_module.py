# -----------------------------------------------------------------------------
# Auteur       : TRISTAN NAULEAU 
# Date         : 2025-07-12
# Licence      : GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
#
# Ce travail est distribué librement en accord avec les termes de la
# GNU GPL v3 (https://www.gnu.org/licenses/gpl-3.0.html).
# Vous êtes libre de redistribuer et de modifier ce code, à condition
# de conserver cette notice et de mentionner que je suis l’auteur
# de tout ou partie du code si vous le réutilisez.
# -----------------------------------------------------------------------------
# Author       : TRISTAN NAULEAU
# Date         : 2025-07-12
# License      : GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
#
# This work is freely distributed under the terms of the
# GNU GPL v3 (https://www.gnu.org/licenses/gpl-3.0.html).
# You are free to redistribute and modify this code, provided that
# you keep this notice and mention that I am the author
# of all or part of the code if you reuse it.
# -----------------------------------------------------------------------------
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter


def calculate_fwhm_ecc(
    data,
    fwhm_guess=3.5,
    threshold_sigma=5.0,
    *,
    sky_bg=None,
    sky_noise=None,
):
    """Calculate median FWHM and eccentricity of stars in ``data``.

    Parameters
    ----------
    data : 2-D numpy array
        Image data.
    fwhm_guess : float
        Initial FWHM guess for DAOStarFinder.
    threshold_sigma : float
        Detection threshold in sigma units.
    sky_bg : float, optional
        Pre-computed sky background level. When finite, this value is used
        instead of recomputing it with ``sigma_clipped_stats``.
    sky_noise : float, optional
        Pre-computed sky noise (standard deviation). When finite and positive,
        this value scales the detection threshold without re-running
        ``sigma_clipped_stats``.

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
        bg = sky_bg if sky_bg is not None and np.isfinite(sky_bg) else None
        noise = (
            sky_noise
            if sky_noise is not None and np.isfinite(sky_noise) and sky_noise > 0
            else None
        )

        if bg is None or noise is None:
            _, median, std = sigma_clipped_stats(data, sigma=3.0)
            if bg is None or not np.isfinite(bg):
                bg = median
            if noise is None or not np.isfinite(noise) or noise <= 0:
                noise = std if std > 0 else np.nan

        if not np.isfinite(bg) or not np.isfinite(noise) or noise <= 0:
            return np.nan, np.nan, 0

        finder = DAOStarFinder(fwhm=fwhm_guess, threshold=threshold_sigma * noise)
        tbl = finder(data - bg)
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
