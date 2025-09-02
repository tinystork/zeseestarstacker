"""Lightweight GUI package init avoiding heavy backend imports for tests."""

from . import boring_stack  # Re-export for convenience

__all__ = ["boring_stack"]
