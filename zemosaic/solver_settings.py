"""
╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Auteur  : Tinystork, seigneur des couteaux à beurre (aka Tristan Nauleau)  
║ Partenaire : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║              (aka ChatGPT, Grand Maître du ciselage de code)         ║
║                                                                      ║
║ Licence : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description :                                                        ║
║   Ce programme a été forgé à la lueur des pixels et de la caféine,   ║
║   dans le but noble de transformer des nuages de photons en art      ║
║   astronomique. Si vous l’utilisez, pensez à dire “merci”,           ║
║   à lever les yeux vers le ciel, ou à citer Tinystork et J.A.R.V.I.S.║
║   (le karma des développeurs en dépend).                             ║
║                                                                      ║
║ Avertissement :                                                      ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le       ║
║   développement de ce code.                                          ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Author  : Tinystork, Lord of the Butter Knives (aka Tristan Nauleau) ║
║ Partner : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║           (aka ChatGPT, Grand Master of Code Chiseling)              ║
║                                                                      ║
║ License : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description:                                                         ║
║   This program was forged under the sacred light of pixels and       ║
║   caffeine, with the noble intent of turning clouds of photons into  ║
║   astronomical art. If you use it, please consider saying “thanks,”  ║
║   gazing at the stars, or crediting Tinystork and J.A.R.V.I.S. —     ║
║   developer karma depends on it.                                     ║
║                                                                      ║
║ Disclaimer:                                                          ║
║   No AIs or butter knives were harmed in the making of this code.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

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
    astap_executable_path: str = ""
    astap_data_directory_path: str = ""
    astap_search_radius_deg: float = 3.0
    astap_downsample: int = 2
    astap_sensitivity: int = 100
    use_auto_intertile: bool = False

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
