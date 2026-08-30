"""Canonical run-scoped registration-reference handoff state.

The original source identity and the temporary prepared FITS are deliberately
separate.  The source path/origin never change; materialization returns a new
descriptor carrying the same canonical identity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class FrozenReference:
    """One immutable reference decision plus its optional prepared artifact."""

    source_path: str
    origin: str
    materialized_path: Optional[str] = None

    def __post_init__(self) -> None:
        raw_source = str(self.source_path).strip()
        if not raw_source:
            raise ValueError("frozen reference source path is empty")
        source = os.path.realpath(os.path.abspath(raw_source))
        if not self.origin:
            raise ValueError("frozen reference origin is empty")
        object.__setattr__(self, "source_path", source)
        if self.materialized_path:
            materialized = os.path.realpath(
                os.path.abspath(str(self.materialized_path))
            )
            object.__setattr__(self, "materialized_path", materialized)

    @property
    def source_basename(self) -> str:
        return os.path.basename(self.source_path)

    def with_materialized(self, path: str) -> "FrozenReference":
        """Return the same frozen decision with a prepared FITS location."""

        return replace(self, materialized_path=path)

    def available_load_path(self) -> Optional[str]:
        """Prefer the canonical source, then its prepared run-local artifact."""

        if os.path.isfile(self.source_path):
            return self.source_path
        if self.materialized_path and os.path.isfile(self.materialized_path):
            return self.materialized_path
        return None
