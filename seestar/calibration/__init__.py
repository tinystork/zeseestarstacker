"""Optional master-calibration integration for ZeSeestarStacker.

This package holds the ZeCalibrator adapter (the only place that imports the
optional ``zecalibrator.api.v1``).  It is deliberately empty at import time:
the adapter is reached lazily, so importing Zsss (or the GUI) never imports
ZeCalibrator, NumPy or Astropy via this path.
"""
