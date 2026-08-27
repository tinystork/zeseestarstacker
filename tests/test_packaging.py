"""Minimal packaging / entry-point checks for ZeSeestarStacker.

These checks are deliberately lightweight and do not import ``seestar`` (which
requires optional deps like OpenCV) nor perform a build. They verify that the
declared version and the ``gui-scripts`` entry point are coherent.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_version_from_init() -> str:
    text = (ROOT / "seestar" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    assert m, "__version__ not found in seestar/__init__.py"
    return m.group(1)


def test_version_is_valid_pep440():
    from packaging.version import Version

    v = _read_version_from_init()
    # Raises if the version is not valid PEP 440.
    Version(v)
    # Exact source-derived check: the declared string is already in canonical
    # PEP 440 form (no normalization drift), not merely non-empty.
    assert Version(v).public == v


def test_pyproject_declares_version_and_entry_point():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]

    assert project["name"] == "ZeSeestarStacker"

    # Dynamic version points at seestar.__version__ (source of truth).
    assert "version" in project.get("dynamic", [])
    assert (
        data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        == "seestar.__version__"
    )

    gui_scripts = project["gui-scripts"]
    assert gui_scripts["zeseestarstacker"] == "seestar.qt_main:main"


def test_pyproject_declares_pyside6_as_standard_dependency():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]
    assert "PySide6" in dependencies


def test_pyproject_has_no_zealfie_dependency():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = data["project"].get("dependencies", [])
    extras = data["project"].get("optional-dependencies", {})
    all_deps = list(dependencies)
    for group in extras.values():
        all_deps.extend(group)
    assert all("zealfie" not in dep.lower() for dep in all_deps)
