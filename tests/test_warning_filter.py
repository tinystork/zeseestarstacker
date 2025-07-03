import importlib.util
import io
import sys
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "drizzle_integration", ROOT / "seestar" / "enhancement" / "drizzle_integration.py"
)
drizzle_integration = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = drizzle_integration
spec.loader.exec_module(drizzle_integration)


def test_kernel_warning_filtered():
    spec.loader.exec_module(drizzle_integration)

    fake_mod = types.ModuleType("drizzle.resample")
    fake_mod.__dict__["__name__"] = "drizzle.resample"
    exec(
        "import warnings\n"
        "def trigger():\n"
        "    warnings.warn(\"Kernel 'whatever' is not a flux-conserving kernel.\", RuntimeWarning)",
        fake_mod.__dict__,
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(
            action="ignore",
            message=r".*is not a flux-conserving kernel\.$",
            module=r"drizzle\.resample",
        )
        fake_mod.trigger()
        fake_mod.trigger()

    assert sum("flux-conserving kernel" in str(rec.message) for rec in w) == 0, "warning not filtered"
