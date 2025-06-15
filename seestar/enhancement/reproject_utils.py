try:
    from reproject.mosaicking import reproject_and_coadd as _reproject_and_coadd
    from reproject import reproject_interp as _reproject_interp
except Exception:  # pragma: no cover - fallback when reproject missing
    def _missing(*_args, **_kwargs):
        raise ImportError(
            "The 'reproject' package is required for this functionality. "
            "Please install it with 'pip install reproject'."
        )
    _reproject_and_coadd = _missing
    _reproject_interp = _missing

reproject_and_coadd = _reproject_and_coadd
reproject_interp = _reproject_interp

__all__ = ["reproject_and_coadd", "reproject_interp"]
