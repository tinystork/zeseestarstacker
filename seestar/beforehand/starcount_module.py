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


def calculate_starcount(data, fwhm=3.5, threshold_sigma=5.0, *, sky_bg=None, sky_noise=None):
    """Return number of stars detected in ``data`` using DAOStarFinder.

    Parameters
    ----------
    data : array-like
        Image data array.
    fwhm : float, optional
        Full width at half maximum expected for stars (pixels).
    threshold_sigma : float, optional
        Detection threshold in multiples of the background noise.
    sky_bg : float, optional
        Pre-computed sky background level. When provided (finite), this value
        is used instead of estimating it again with ``sigma_clipped_stats``.
    sky_noise : float, optional
        Pre-computed sky noise (standard deviation). When provided (finite and
        positive), this value is used to scale the detection threshold.
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
            return 0

        finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * noise)
        sources = finder(data - bg)
        return 0 if sources is None else len(sources)
    except Exception:
        return 0

