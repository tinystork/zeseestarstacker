# ZeSolver RGB WCS Canonicalization — Final Closure

Status: CLOSED
Release: ZeSeestarStacker 8.2.2 — Phoenix consedit
Fix commit: d415d8556f5bc94faeb7b3b7ffcf85e867301cda
Release commit: a63d6d0576be53cbc42dc78ee3dce4384e22dc45

## Root cause

ZeSolver correctly solved the RGB reference FITS cube (`NAXIS=3`).

The returned WCS contained a valid celestial RA/DEC component:

- `has_celestial = True`
- `is_celestial = False`

because Astropy considered the full three-axis WCS not to be a purely
celestial 2D WCS.

ZSSS incorrectly used the latter condition as an acceptance gate and
therefore treated a successful ZeSolver result as a failure, triggering
ASTAP or approximate-WCS fallback.

## Fix

The ZeSolver adapter now canonicalizes a successful result to its
two-dimensional celestial sub-WCS (`wcs.celestial`).

A SOLVED result without a usable celestial component is still rejected
explicitly.

No ZeSolver, ZeAlfie, stacking, Drizzle, or RF2 architecture changes were
required.

## Regression witness

Before:

    NAXIS=3
    CTYPE = RA---TAN / DEC--TAN / empty
    has_celestial = True
    is_celestial = False
    -> false ASTAP fallback

After:

    pixel_n_dim = 2
    world_n_dim = 2
    CTYPE = RA---TAN / DEC--TAN
    is_celestial = True
    -> accepted by AstrometrySolver

Threading was demonstrated not to be causal.

## E2E closure

Linux:
- 20 deliberately WCS-free M16 frames
- ZeSolver 1.2.1 / API 1.2
- ZeSolver-only solve
- no ASTAP fallback
- real M16 celestial grid recovered
- FINISHED

Windows / ZeAlfie shared runtime:
- ZeSeestarStacker 8.2.2
- 20 deliberately WCS-free M16 frames
- reference solved by ZeSolver
- reference scale ~2.37 arcsec/pixel
- frozen reference WCS reused
- 20/20 aligned
- final FITS saved
- RUN_SUCCEEDED

## Provenance

Source bundle:

    ZeSeestarStacker_source_8.2.2_a63d6d0.zip

SHA256:

    9869e9b68bd9aae0a15ac6dbc26412fbce4b72cebc9683c1f8abf1374fe50812

Full closure evidence is retained externally as:

    ZSSS_8.2.2_ZESOLVER_WCS_CANONICAL_FINAL_CLOSURE_a63d6d0.zip

See:

    docs/witnesses/zesolver_wcs_canonical_8.2.2/