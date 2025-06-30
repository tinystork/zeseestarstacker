import numpy as np

try:
    import cupy as cp  # type: ignore
    _cupy_available = cp.cuda.is_available()
except Exception:
    cp = None  # type: ignore
    _cupy_available = False


def drizzle_finalize(
    sci_sum: np.ndarray,
    wht_sum: np.ndarray,
    mode: str = "divide",
    use_gpu: bool | None = None,
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
    if use_gpu is None:
        use_gpu = _cupy_available

    xp = cp if use_gpu and _cupy_available else np

    sci = xp.asarray(sci_sum, dtype=xp.float32)
    wht = xp.asarray(wht_sum, dtype=xp.float32)
    if mode not in {"divide", "none", "max", "n_images"}:
        mode = "divide"
    if mode == "none":
        result = sci
    else:
        wht_safe = xp.maximum(wht, 1e-9)
        result = sci / wht_safe
        if mode == "max":
            result *= float(xp.max(wht_safe))
        elif mode == "n_images":
            result *= float(xp.mean(wht_safe))

    result = xp.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(xp.float32)
    if xp is cp:
        result = cp.asnumpy(result)
    return result.astype(np.float32)
