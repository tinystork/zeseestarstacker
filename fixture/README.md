# Fixture: immutable master reference dataset

This directory holds the **immutable** master reference dataset used by the
reliability audit (mission ZSSS-QT-RELIABILITY-AUDIT).  The same FITS files
are fed to Tk, Qt, and the engine so any behavioural difference is a *real*
difference — not a difference in inputs.

## Layout

- `master/` — the generated, **never-modified** master images
  (`Light_001.fit` .. `Light_010.fit`).
- `generate_master.py` — deterministic generator (fixed seed, reproducible).
- `copy_fresh.py` — helper to copy fresh images out of `master/` into a
  scratch/tmp directory.

## Master spec

| Property | Value |
|----------|-------|
| Sensor   | ~1080 x 1920 (H x W), Seestar S50 geometry |
| Pixel scale | ~2.37 arcsec / px (CDELT) |
| Projection | RA---TAN / DEC--TAN, centred on M16 (Eagle Nebula) |
| Data type | 16-bit unsigned integer (Seestar-style RAW) |
| Noise | none (smooth, deterministic, compressible) |
| Count    | 10 images, `Light_001.fit` .. `Light_010.fit` |
| Seed     | `20260821` (fixed) |

## Regenerating

```bash
python fixture/generate_master.py
```

The master files are **never edited by hand**.  If the spec must change, bump
the seed/version and regenerate explicitly (and update this README).

## Copying fresh images (tests / manual runs)

```python
from fixture.copy_fresh import copy_master_to
copied = copy_master_to(tmp_path)      # all 10
copied = copy_master_to(tmp_path, n=4) # first 4
```
