## Unreleased
### Fixed
* Robust FITS writing for ASTAP CLI (colour order normalised)
### Added
* Solver option “Convert to Luminance” to force mono before plate-solve
* Added configurable `winsor_worker_limit` (CLI `--winsor-workers` / `-W` and GUI field)
* Added `winsor_max_frames_per_pass` streaming limit for Winsorized rejection (GUI/config)
* Manual frame cap via `max_raw_per_master_tile` (CLI/GUI/config)
* Fixed incremental assembly with reproject>=0.11
* 16-bit FITS export now writes a 2D luminance primary with R/G/B extensions for broader viewer compatibility, with a legacy RGB cube available from Advanced options
