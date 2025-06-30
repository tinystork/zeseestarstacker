import numpy as np


def drizzle_finalize(
    sci_sum: np.ndarray, wht_sum: np.ndarray, mode: str = "divide"
) -> np.ndarray:
    """Finalize drizzled data by normalising once.

    Parameters
    ----------
    sci_sum : np.ndarray
        Accumulated science array (flux * weight).
    wht_sum : np.ndarray
        Accumulated weight array.
    mode : str, optional
        Normalisation mode: ``"divide"`` (default), ``"none"``, ``"max``", or
        ``"n_images"``.

    Returns
    -------
    np.ndarray
        Normalised image as ``float32`` with invalid values replaced by ``0``.
    """
    sci = np.asarray(sci_sum, dtype=np.float32)
    wht = np.asarray(wht_sum, dtype=np.float32)
    if mode not in {"divide", "none", "max", "n_images"}:
        mode = "divide"
    if mode == "none":
        result = sci
    else:
        wht_safe = np.maximum(wht, 1e-9)
        result = sci / wht_safe
        if mode == "max":
            result *= float(np.max(wht_safe))
        elif mode == "n_images":
            result *= float(np.mean(wht_safe))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
