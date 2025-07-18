import os
import sys

# Ensure the top-level package is importable during tests
_parent = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
