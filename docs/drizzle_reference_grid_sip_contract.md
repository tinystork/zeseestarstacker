# Drizzle reference-grid SIP contract

Mission: `zsss-drizzle-reference-grid-closure-20260913` (rework-3). Scope:
exact SIP geometry support for the Standard reference-anchored output grid.

## Cause (physical witness failure)

A frozen USER reference solved by ASTAP carries a TAN-**SIP** WCS. The canonical
builder `seestar.core.drizzle_core.build_output_grid` previously refused *any*
SIP (`unsupported WCS distortion for output grid: sip`), which aborted
accumulator initialization on the Windows Standard scale-2 Lanczos2 witness
even though manual reference authority and materialization were correct.

Root cause: a blanket SIP rejection added as a safety guard. Production
`pixmap_from_alignment` already maps through `all_pix2world`/`all_world2pix`,
so SIP is fully usable as long as the output grid carries an exactly-scaled SIP.

## Exact SIP contract

For a grid transform `p_out = s*(p_ref + 0.5) - 0.5` (FITS edge/centre
preserving, zero-based), let `u = p - crpix` (with the **SIP** origin, not the
linear CRPIX). SIP forward/inverse polynomials satisfy, for both `A`/`B` and
`AP`/`BP`:

```
C_out[i, j] = C_ref[i, j] * s ** (1 - i - j)
```

because `f_out(u_out, v_out) = s * f_ref(u_out/s, v_out/s)` with
`u_out = s * u_ref`. Same scaling for the optional inverse AP/BP.

Properties preserved by `build_output_grid`:

- effective pixel-scale matrix `M_out = M_ref / s` (real `CD` scaled, else
  `CDELT` scaled with `PC` preserved);
- `CRVAL`, `CTYPE`, projection, celestial frame, orientation, handedness;
- `CRPIX_out = s*(CRPIX_ref - 0.5) + 0.5`;
- SIP origin transformed with the same formula (independent copy);
- forward-only SIP stays forward-only; a real inverse AP/BP is scaled too;
- public `array_shape`/`pixel_shape` equal to the returned shape;
- the reference WCS is never mutated.

Numerically validated: max mapping residual `< 1e-9` px for scales 1–4,
non-square rotated TAN-SIP, off-centre CRPIX, `CD` and `PC+CDELT`, forward-only
and inverse-bearing.

### Refusals (fail closed, never silently dropped)

- lookup-table (`CPDIS`) and detector-to-image (`det2im`) distortions;
- `has_distortion` without SIP;
- a SIP origin that differs from the linear CRPIX (FITS cannot persist it);
- mismatched forward/inverse orders; non-finite coefficients.

Solver configuration and the science (deposition, kernels, pixfrac, PSR, SCI/
WHT/SUP, weighting, histogram/render) are untouched.

## Persistence / identity

- `serialize_wcs_header` persists SIP `A/B/AP/BP` cards via
  `to_header(relax=True)`. A forward-only SIP is persisted without the empty
  `AP_ORDER/BP_ORDER` cards so the header round-trips exactly.
- Astropy 8.0.1 drops the SIP *inverse* on header re-parse, so
  `_sip_from_cards` reconstructs the full SIP explicitly (forward and inverse)
  in both output-WCS and input-reference-geometry restore paths. Round-trips
  are exact and mapping-preserving.
- The full-grid identity snapshot includes, for input and output: SIP presence,
  origin, A/B/AP/BP orders and every coefficient, so a same-PSR SIP
  mutation/add/remove fails closed; repeated equivalent resolution stays
  idempotent.

## Current state / tests

- `tests/test_reference_grid_closure_sip.py` (gate coverage): exact scaled SIP
  copy (PC/CD, scales 1–4, forward-only/inverse), scale-1 identity, perimeter
  and centre containment via production `pixmap_from_alignment`, tiny
  deposition, lookup/det2im refusal, SIP-origin-mismatch refusal, full-grid
  identity drift, checkpoint write/read/**continue** with SIP (uninterrupted vs
  stop/resume bit-identical), and a real `start_processing`
  TAN-SIP-reference→initialized-accumulators test (USER, zero AUTO).
- `tests/test_reference_grid_closure_phase2.py::test_lookup_table_distortion_is_refused`
  supersedes the old SIP-refusal assertion.
- Non-SIP checkpoint/resume/reliability suites remain green; v1 legacy grid
  refusal and v2 non-SIP compatibility unchanged.
