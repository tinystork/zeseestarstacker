import importlib.util
import io
import sys
import warnings
import contextlib
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "drizzle_integration", ROOT / "seestar" / "enhancement" / "drizzle_integration.py"
)
drizzle_integration = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = drizzle_integration
spec.loader.exec_module(drizzle_integration)
sys.path.insert(0, str(ROOT / "seestar" / "enhancement"))


def test_kernel_warning_is_filtered():
    # reload to ensure filter is set
    import drizzle_integration as di
    importlib.reload(di)

    fake_mod = types.ModuleType("drizzle.resample")
    exec(
        "import warnings\n"
        "def emit():\n"
        "    warnings.warn(\"Kernel 'whatever' is not a flux-conserving kernel.\", RuntimeWarning)",
        fake_mod.__dict__,
    )

    captured = io.StringIO()
    with contextlib.redirect_stderr(captured):
        fake_mod.emit()
    assert captured.getvalue() == "", "Warning was not filtered!"
