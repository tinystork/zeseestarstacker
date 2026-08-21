# ZSSS QT PREFLIGHT — Mission board

Repo: `~/projects/zeseestarstacker` — branch `feature/pyside6-migration`
HEAD final: `8a68497` (ZSSS-QT-PREFLIGHT-CLOSURE) — commit local autorisé, NO PUSH/NO MERGE/NO TAG/NO M26.
Verdict: **QT_PREFLIGHT_READY** — witness humain final requis (procédure courte fournie).

## Statut final

- [x] Baseline git établi (HEAD b6ac30d)
- [x] D3 (Coco + review): Scale/WHT/Kernel/Pixfrac dans le bloc Drizzle de Stacking, ordre Tk exact, Expert nettoyé
- [x] Lifecycle (Coco + review + correctif progress): SUCCESS/FAILED/CANCELLED/EMPTY-NO OUTPUT distincts
- [x] Closure A (Coco + review): summary utilise `final_stacked_path` réel (source de vérité), plus de faux EMPTY sur vrais produits; boring garde fallback documenté (copie vers final.fits); tests multi-noms de sortie
- [x] Closure B (Coco + review + correctif sys.modules): `move_stacked=True` par défaut restauré (checkpoint filesystem historique); False = mode sécurité explicite; callers tous en chemins de succès; tests 13/13 + arrêt partiel
- [x] Closure C (Coco + review): label "Preview group size", hint M3 supprimé de l'UI Qt
- [x] Versioning (Coco + review): 8.0.0 "Phoenix consedit" partout (__init__, version string runtime, tests, checklist, CHANGELOG.md)
- [x] Correctif review: tests reliability restaurent sys.modules (stubs ne polluent plus la collecte)
- [x] Commit local `8a68497` (21 fichiers, +1068/-142), fixture/audit_drizzle_scale.py laissé untracked (artefact temporaire)

## Validation

- Suite reliability + Qt: **579 passed / 1 échec pré-existant** (test_tk_gui_still_imports, artefact TkAgg headless)
- `git diff --check` propre
- Collecte complète `tests/`: 10 erreurs + 24 échecs **identiques au HEAD b6ac30d** (interaction stubs legacy) — zéro régression introduite
- Matrice scientifique (missions précédentes): 6/6 workflows FINISHED, shapes/scale corrects, parité kwargs Tk=Qt

## À faire par Tristan (witness final)

Voir `docs/qt_reliability_witness_procedure.md` : Drizzle Standard ×2, un mode Classic/Reproject, Stop partiel pour stacked/ — vérifier output réel reconnu, SUCCESS correct, stats résumé, filename correct, fichiers consommés dans stacked/, preview OK.

## Décisions produit (ne pas rediscuter)

- GPU/Language/Theme → onglet System
- D3: scale/WHT/kernel/pixfrac = bloc Drizzle principal de Stacking
- Final/Incremental = internes; labels = Standard / Large dataset
- stacked/ = checkpoint filesystem historique (move_stacked=True par défaut)
- M26 reste bloqué jusqu'au witness installé final; pas de push/merge/tag
