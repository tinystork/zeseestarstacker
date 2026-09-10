# P2-A physical witness — exact bounded human-run recipe (rework-2)

This recipe captures real per-add deposition truth at the Phase-1 physical
pixels. It is **diagnostic only** and opt-in: with the env var unset, the run
is byte-for-byte the normal production run.

Three kernel runs may be executed **sequentially inside one GUI process**:
the witness binds a fresh, run-scoped recorder at every run start (canonical
new-run seam), writes to each run's own output folder, and never resets
mid-run. Do **not** set `ZSSS_DEPOSITION_TRUTH_OUT` (that constant path would
let later runs overwrite earlier artifacts).

## 1. Exact launcher (single command)

From the repository root:

    cd /home/tristan/.openclaw/workspace/projects/zeseestarstacker
    ZSSS_DEPOSITION_TRUTH_TARGETS="$PWD/research/drizzle_scientific_closure_p2/p2a/targets_phase1.json" \
      .venv/bin/python -m seestar.qt_main

(No secrets are involved.) `ZSSS_DEPOSITION_TRUTH_OUT` must stay unset.

## 2. GUI settings (identical to the Phase-1 runs)

- Input folder: `/home/tristan/Téléchargements/test drizzle` (the same
  40-image source set).
- Reference: `Light_Unknown_20.0s_LP_20260823-035631.fit` (or the same
  `AUTO_GEOMETRY` selection used in Phase 1).
- Stacking: **Drizzle** (Standard M3, direct accumulation).
- Scale: **3**; Pixfrac: **1**; WHT threshold: **0**.
- Weighting: `noise_variance` with quality weighting / SNR / stars ON.
- Save as float32: ON.
- Normalisation: `sky_mean`.

## 3. Sequence (one process, three runs)

For kernel ∈ {`lanczos2`, `lanczos3`, `square`}, in this order:

1. set Kernel = the current kernel, Kernel-relative WHT threshold = 0;
2. set the **Output folder** to a fresh, distinct directory, e.g.
   `/home/tristan/Téléchargements/out/p2a_witness_<kernel>` (never the Phase-1
   folders);
3. start the stack and let it finish;
4. collect the artifact `<output folder>/drizzle_deposition_truth.json`.

No restart is required between runs, and no shell env change is required
between runs — only the GUI Kernel and Output folder change.

Expected artifact provenance per run:
`provenance.effective_kernel` = the kernel actually used;
`provenance.skipped_by_kernel` counts targets belonging to the other kernels.

## 4. What the artifact proves

Per target: `sum_D`, `pos_D`, `neg_D`, `sum_abs_D`, `sum_N`, `pos_N`,
`neg_N`, `sum_abs_N`, `cancellation_quality`, `reconstructed_sci`,
`initial_native_img/wht`, `final_native_img/wht`,
`closure_sum_D_err`, `closure_sum_N_err`, `initial_state_zero`,
`resume_nonzero_initial`; plus bounded audited per-add `rows` and the honest
`measured_state_bytes`.

Closure is checked **per target** as `sum_D - (final_wht - initial_wht)` and
`sum_N - (final_numerator - initial_numerator)`. A fresh run must show
`initial_state_zero = true`; a resumed run is explicitly labelled
`resume_nonzero_initial = true`.

## 5. Escalation (only if needed)

If the per-add net deltas do **not** expose both signs at a catastrophic
target, escalate within this bounded phase to a finer selected-pixel
installed-engine decomposition (per-input-pixel signed weights). Never infer
cancellation from aggregate values alone.
