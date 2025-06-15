
"""Initialize zemosaic package and expose modules with legacy names."""

from importlib import import_module
import sys

for _name in [
    "zemosaic_utils",
    "zemosaic_astrometry",
    "zemosaic_align_stack",
    "zemosaic_config",
]:
    try:
        mod = import_module(f".{_name}", __name__)
        sys.modules[_name] = mod
    except Exception:
        pass
