from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

DEFAULT_SETTINGS_FILE = Path(__file__).with_name("solver_settings.json")


@dataclass
class SolverSettings:
    """Persistent plate solver parameters."""

    solver_choice: str = "ASTAP"
    api_key: str = ""
    timeout: int = 60
    downsample: int = 2
    force_lum: bool = False

    @staticmethod
    def default_path() -> Path:
        """Return the default settings file path."""
        return DEFAULT_SETTINGS_FILE

    def save(self, path: str | Path) -> None:
        """Save settings to a JSON file."""
        p = Path(path)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    def save_default(self) -> None:
        """Save settings to the default file."""
        self.save(self.default_path())

    @classmethod
    def load(cls, path: str | Path) -> "SolverSettings":
        """Load settings from a JSON file."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)

    @classmethod
    def load_default(cls) -> "SolverSettings":
        """Load settings from the default file if present."""
        p = cls.default_path()
        if p.exists():
            return cls.load(p)
        return cls()
