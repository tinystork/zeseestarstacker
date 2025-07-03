## Unreleased
### Fixed
* Robust FITS writing for ASTAP CLI (colour order normalised)
### Added
* Solver option “Convert to Luminance” to force mono before plate-solve
* Added configurable `winsor_worker_limit` (CLI `--winsor-workers` / `-W` and GUI field)
* Manual frame cap via `max_raw_per_master_tile` (CLI/GUI/config)
* Fixed incremental assembly with reproject>=0.11
