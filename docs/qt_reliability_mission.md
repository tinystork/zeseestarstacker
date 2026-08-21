# ZSSS QT RELIABILITY — Mission board

Repo: `~/projects/zeseestarstacker` — branch `feature/pyside6-migration`
HEAD au début de mission: `b6ac30d` (ZSSS-QT-RELIABILITY-R3b)
Oracle: GUI Tk même HEAD (captures fournies) + moteur réel
Règles: NO PUSH / NO MERGE / NO TAG / NO M26 — parité Tk d'abord, ergonomie ensuite.
Objectif actuel: **QT_PREFLIGHT_READY** (closure A/B/C + versioning 8.0.0, commit local autorisé à la fin).

## Statut

- [x] Baseline git établi (HEAD b6ac30d)
- [x] Suite Qt complète: 564 passed / 1 échec pré-existant (test_tk_gui_still_imports, artefact TkAgg headless)
- [x] R1/R2/R3a/R3b vérifiés
- [x] **D3** (Coco + review Jarvis ACCEPTÉE): contrôles Drizzle dans bloc Stacking, ordre Tk
- [x] **Lifecycle** (Coco + review Jarvis ACCEPTÉE + correctif progress): SUCCESS/FAILED/CANCELLED/EMPTY-NO OUTPUT
- [x] **Matrice Tk vs Qt** (vraies M16): 6/6 FINISHED, shapes/scale corrects, zéro mutation (sous l'ancien invariant), parité kwargs 6/6
- [x] **Closure A — summary final path** (Coco + review Jarvis ACCEPTÉE): build_summary_payload accepte final_stack_path (source de vérité = stacker.final_stacked_path), fallback legacy final.fits, backend/boring branchés, tests paramétrés sur 4 vrais noms (final.fits, stack_final_classic_sumw, stack_final_drizzle_final, stack_final_classic_reproject)
- [x] **Closure B — stacked/ restore** (Coco + review Jarvis ACCEPTÉE): move_stacked=True par défaut (2 inits + start_processing), gate R1 conservé comme mode sécurité explicite False, callers vérifiés tous en chemins de succès, tests R1 adaptés + nouveau test_reliability_stacked_restore_b.py (5 tests: défaut, move, False zéro-mutation, 13/13 witness, arrêt partiel)
  - Vérification indépendante: échecs engine signalés = pré-existants (confirmé sur worktree HEAD propre, ex: test_save_final_stack)
- [ ] **Closure C — UI drizzle** (en cours chez Coco): "Drizzle group size" → "Preview group size" (présentation), suppression hint M3 de l'UI Qt
- [ ] **Versioning 8.0.0 Phoenix consedit** (en cours chez Coco): __version__/__codename__, constante runtime, tests, CHANGELOG.md, docs — pas de tag/push
- [ ] Witness ciblé final (Drizzle Std ×2, Classic/Reproject, Stop partiel stacked/)
- [ ] Commit local propre (autorisé par Tristan après closure validée) — NO PUSH/NO MERGE/NO TAG
- [ ] Verdict **QT_PREFLIGHT_READY** + procédure dernier witness + STOP

## Décisions produit (ne pas rediscuter)

- GPU → onglet System ; Language → System ; Theme System/Dark/Light → System
- D3: scale/WHT/kernel/pixfrac = bloc Drizzle principal de Stacking, pas Expert, pas dupliqués (FAIT)
- Final/Incremental = valeurs internes, labels utilisateur = Standard / Large dataset
- Standard et Large dataset = même science, politique de ressources différente
- **stacked/**: déplacement après consommation réussie = comportement historique voulu (checkpoint filesystem); move_stacked=False = mode sécurité explicite (tests/harness) — PAS le défaut
- **Summary**: chemin du produit final réel (final_stacked_path) = source de vérité; pas de liste fragile de noms dans le GUI

## Journal

- 2026-08-21 22:45 — Début mission (handoff Tristan). Baseline établi.
- 2026-08-21 ~23:15 — D3 review ACCEPTÉE (550 passed).
- 2026-08-21 ~23:40 — Matrice réelle 6/6 FINISHED (vraies M16). Découverte TMPDIR/SIGBUS.
- 2026-08-21 ~23:55 — Lifecycle review ACCEPTÉE (558 passed) + correctif progress reset.
- 2026-08-22 00:36 — Nouveau handoff Tristan: closure A (summary path), B (stacked/ restore), C (UI drizzle), versioning 8.0.0, commit local autorisé à la fin, QT_PREFLIGHT_READY.
- 2026-08-22 ~01:00 — Closure A review ACCEPTÉE (tests 4 vrais noms, 29 passed ciblés).
- 2026-08-22 ~01:05 — Closure B review ACCEPTÉE (564 passed, 11 tests reliability, échecs engine pré-existants confirmés sur worktree HEAD propre).
- 2026-08-22 ~01:10 — Closure C + versioning 8.0.0 en cours chez Coco.
