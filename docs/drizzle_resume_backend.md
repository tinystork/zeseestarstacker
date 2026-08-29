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
