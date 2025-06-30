from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class SolverSettings:
    """Persistent plate solver parameters."""

    solver_choice: str = "ASTAP"
    api_key: str = ""
    timeout: int = 60
    downsample: int = 2

    def save(self, path: str | Path) -> None:
        """Save settings to a JSON file."""
        p = Path(path)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SolverSettings":
        """Load settings from a JSON file."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)
