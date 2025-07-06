import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


def calculate_starcount(data, fwhm=3.5, threshold_sigma=5.0):
    """Return number of stars detected in ``data`` using DAOStarFinder."""
    try:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
        sources = finder(data - median)
        return 0 if sources is None else len(sources)
    except Exception:
        return 0

