"""Version consistency: the product display version is derived from the package
source of truth (``seestar.__version__`` + ``seestar.__codename__``) and is no
longer hardcoded as ``6.3.0 Boring``.

These tests avoid importing the heavy ``seestar`` package (OpenCV / GUI are not
available in this environment) by:

* reading ``seestar/__init__.py`` for the source-of-truth strings,
* loading ``seestar.gui.settings`` standalone (stdlib + numpy + tkinter only)
  WITHOUT preloading a fake ``seestar`` package, so the source-tree fallback is
  exercised exactly like a plain source checkout where ``import seestar`` fails
  because optional deps (e.g. OpenCV) are missing,
* checking ``seestar.queuep.queue_manager`` statically via AST.

The expected values are read from the source rather than hardcoded, so these
tests stay branch-agnostic and cannot drift when the canonical version changes.
"""

import ast
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_package_metadata():
    text = (ROOT / "seestar" / "__init__.py").read_text(encoding="utf-8")
    version = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE
    )
    codename = re.search(
        r'^__codename__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE
    )
    assert version, "__version__ not found in seestar/__init__.py"
    assert codename, "__codename__ not found in seestar/__init__.py"
    return version.group(1), codename.group(1)


def _expected_display_version():
    version, codename = _read_package_metadata()
    return f"{version} {codename}"


def _restore_module(name, prev):
    if prev is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = prev


# --- settings.py standalone loading (mirrors tests/test_m3d_settings.py) ---


def _load_settings_module():
    """Load ``seestar.gui.settings`` standalone, WITHOUT preloading a fake
    ``seestar`` package. This matches a plain source checkout: the only
    ``import seestar`` in settings.py is lazy (inside ``_product_display_version``)
    and may legitimately fail when optional deps such as OpenCV are missing, so
    the helper must fall back to the source tree."""
    spec = importlib.util.spec_from_file_location(
        "seestar.gui.settings_standalone",
        ROOT / "seestar" / "gui" / "settings.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_package_source_of_truth():
    version, codename = _read_package_metadata()
    assert version
    assert codename
    # Display version must be exactly "__version__ __codename__".
    assert _expected_display_version() == f"{version} {codename}"


def test_settings_save_writes_derived_display_version(tmp_path):
    version, codename = _read_package_metadata()
    expected = f"{version} {codename}"

    # Do NOT mask the failure: drop any previously injected fake ``seestar`` so
    # that ``import seestar`` inside the helper fails naturally in this env.
    saved_seestar = sys.modules.pop("seestar", None)
    try:
        mod = _load_settings_module()

        out = tmp_path / "seestar_settings.json"
        sm = mod.SettingsManager(settings_file=str(out))
        sm.save_settings()

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["version"], "saved JSON version must be non-empty"
        assert data["version"] == expected
        assert data["version"] != "6.3.0 Boring"
    finally:
        _restore_module("seestar", saved_seestar)


def test_settings_load_old_version_still_succeeds(tmp_path):
    mod = _load_settings_module()

    out = tmp_path / "old_settings.json"
    # A legacy settings file carrying the stale hardcoded version. ``version``
    # is not consumed as a schema/migration version, so it must be ignored.
    out.write_text(
        json.dumps({"version": "6.3.0 Boring", "kappa": 3.2, "batch_size": 7}),
        encoding="utf-8",
    )

    sm = mod.SettingsManager(settings_file=str(out))
    assert sm.load_settings() is True
    # Known settings still load normally alongside the ignored version key.
    assert sm.kappa == 3.2
    assert sm.batch_size == 7


def test_queue_manager_version_string_derived_from_source():
    version, codename = _read_package_metadata()
    expected = f"{version} {codename}"
    src = (ROOT / "seestar" / "queuep" / "queue_manager.py").read_text(
        encoding="utf-8"
    )

    tree = ast.parse(src)
    helper_name = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "GLOBAL_DRZ_BATCH_VERSION_STRING_ULTRA_DEBUG"
            ):
                # The global must be assigned from a helper call, not a
                # hardcoded release display literal.
                assert isinstance(node.value, ast.Call), (
                    "GLOBAL_DRZ_BATCH_VERSION_STRING_ULTRA_DEBUG must be "
                    "assigned from a helper call, not a hardcoded literal"
                )
                func = node.value.func
                assert isinstance(func, ast.Name), (
                    "helper must be a plain function call"
                )
                helper_name = func.id
    assert helper_name, "GLOBAL_DRZ_BATCH_VERSION_STRING_ULTRA_DEBUG not found"

    # No stale hardcoded literal assignment may remain anywhere.
    assert not re.search(
        r"GLOBAL_DRZ_BATCH_VERSION_STRING_ULTRA_DEBUG\s*=\s*[\"']",
        src,
    ), "stale hardcoded literal assignment remains"

    helper_node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == helper_name
    )
    helper_src = ast.get_source_segment(src, helper_node)

    # Evaluate the helper in isolation. ``__file__`` lets the source-tree
    # fallback locate ``seestar/__init__.py`` when the package is not imported.
    ns = {"__file__": str(ROOT / "seestar" / "queuep" / "queue_manager.py")}
    exec(compile(helper_src, "<queue_manager-helper>", "exec"), ns)

    # Fast path: package attributes available.
    fake_seestar = types.ModuleType("seestar")
    fake_seestar.__version__ = version
    fake_seestar.__codename__ = codename
    prev = sys.modules.get("seestar")
    sys.modules["seestar"] = fake_seestar
    try:
        assert ns[helper_name]() == expected
    finally:
        _restore_module("seestar", prev)

    # Fallback path: no importable ``seestar`` — must derive from source tree.
    prev = sys.modules.pop("seestar", None)
    try:
        assert ns[helper_name]() == expected
    finally:
        _restore_module("seestar", prev)
