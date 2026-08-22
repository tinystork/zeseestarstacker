# Changelog

All notable changes to ZeSeestarStacker are documented in this file.

## [7.1.1] - 2026-08-22

### Fixed

- Version consistency: the product display version (`seestar.gui.settings` save
  and the DRZ batch debug string) is now derived from the package source of
  truth (`seestar.__version__` + `seestar.__codename__`) instead of a
  hardcoded literal.
