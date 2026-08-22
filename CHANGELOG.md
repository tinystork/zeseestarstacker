# Changelog

All notable changes to ZeSeestarStacker are documented in this file.

## [8.0.0] — Phoenix consedit

- PySide6 GUI replaces Tkinter as the primary interface
- Tkinter retained temporarily as explicit fallback
- Drizzle UI parity restored
- Qt backend lifecycle and processing summary hardened
- settings moved to platform user-data paths
- System tab / theme / language integration

## [7.1.1] - 2026-08-22

### Fixed

- Version consistency: the product display version (`seestar.gui.settings` save
  and the DRZ batch debug string) is now derived from the package source of
  truth (`seestar.__version__` + `seestar.__codename__`) instead of a
  hardcoded literal.
