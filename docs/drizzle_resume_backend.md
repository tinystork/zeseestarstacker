# Native Drizzle resume (backend/headless)

Standard non-mosaic Drizzle runs can resume only from the dedicated
`.m3d_checkpoint` state and only when Resume intent is explicit. Mosaic and
reprojection modes remain unsupported.

Startup validates the complete checkpoint, exact NumPy/Drizzle versions,
runtime-effective scientific configuration, persisted WCS/grid, counters,
ordered completed ledger, reference, input roots, and exact remaining source
suffix before the worker starts. Validation is read-only and failure preserves
the last committed generation.

Source lookup is strict by default. When `move_stacked` is enabled, the only
additional accepted location is the deterministic `stacked/<original basename>`
counterpart, verified again by size and `mtime_ns`; arbitrary renames and
collision suffixes are refused.

Continuation always performs a fresh disk re-read and restores the three native
SCI/WHT accumulators. It re-arms the writer at generation N+1, preserving signed
Lanczos weights and cumulative exposure/counter truth. Fresh Drizzle and classic
SUM/WHT behavior are unchanged.

## Explicit Qt readiness

The Qt **Resume** selector recognizes standard non-mosaic Drizzle runs through
the pair `<run>/.m3d_checkpoint/checkpoint.json` and `<run>/run_config.cfg`.
This is only a read-only readiness layer: it verifies the checkpoint identity
and the canonical config digest/fingerprint, while the backend still performs
the complete authoritative validation described above.

On readiness, Qt builds an explicit Drizzle request with `resume_intent=resume`,
`resume_source=<run>`, `use_drizzle=True`, and mosaic/reprojection disabled.
Because the checkpoint CFG stores the runtime-effective deposition contract,
the scale, kernel, pixfrac, and WHT threshold controls are restored from the
`*_effective` fields. This includes restoring Lanczos pixfrac as `1.0`; the UI
does not invent or claim an unavailable original requested value.

Drizzle never falls back to a legacy CFG. A missing/corrupt/mismatched CFG or
manifest leaves the selector Fresh. If Classic and Drizzle manifests coexist in
one run directory, Qt refuses the ambiguous run rather than choosing one.
