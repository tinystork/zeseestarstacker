# Mission M3-D — Large dataset / incremental Drizzle

Statut : **EN COURS** — M3-D-1 et M3-D-2 ACCEPTÉES, M3-D-3 implémentée (revue en attente) (2026-08-20).
Source : mandat Tristan (2026-08-20). Règles projet : aucun push, aucun run long répété, tests synthétiques d'abord.

## Définition scientifique (invariant principal)

Le Drizzle M3 est défini par **un accumulateur unique** alimenté avec les poses
originales, leurs transformations et leurs poids. Le batch/group est une unité
de gestion mémoire, progression, preview et (éventuellement) checkpoint/reprise.
**Il n'est jamais une unité scientifique de combinaison.**

Pour les mêmes poses, transformations, poids et paramètres Drizzle :

```
Final
Incremental
group_size = 2
group_size = 20
group_size = 200
```

doivent produire le **même SCI**, le **même WHT** et le **même WCS final**
(tolérance numérique définie).

## Architecture cible

```
poses originales
 ↓
lecture par groupes
 ↓
transformations + poids
 ↓
MÊME accumulateur Drizzle M3 (drizzle_accumulators, 3 canaux)
 ↓
libération du groupe
 ↓
groupe suivant
 ↓
finalize une seule fois (DrizzleAccumulator.finalize)
 ↓
SCI / WHT / WCS final
```

- `DrizzleAccumulator` (seestar/core/drizzle_core.py) reste la source de vérité.
- **Ne pas** restaurer l'ancien modèle (lots → images Drizzle intermédiaires →
  re-stack/re-drizzle). `_process_incremental_drizzle_batch`, objets
  `incremental_drizzle_objects`, `_save_drizzle_input_temp`,
  `_start_drizzle_process`, `drizzle_batch_worker` sont de l'historique invalidé.
- La distinction `standard`/`incremental` est une **politique de traitement**
  (`drizzle_processing_policy`), pas deux sciences différentes. Pas de
  `FINALIZATION_MODE_DRIZZLE_FINAL` / `..._INCREMENTAL` : une seule logique de
  finalisation (celle de `_save_final_stack`, branchée sur
  `FINALIZATION_MODE_DRIZZLE`).
- Preview intermédiaire : artefact d'affichage dérivé de l'accumulateur
  (ex. SCI/WHT → stretch/downsample), **jamais réinjecté** dans le calcul.
- Mémoire : empreinte de l'accumulateur indépendante du nombre de poses
  (2 tableaux float32 HxW par canal). Ne pas créer de mécanisme plus complexe
  pour reproduire l'ancien Incremental si c'est déjà borné.
- Checkpoint : étude séparée (design only), préserve l'état mathématique de
  l'accumulateur (SUM/WHT/méta/index), pas une image intermédiaire.

## Baseline vérifiée (HEAD bc87eb3, branche beta, 2026-08-20)

- M3-C en place : worker → `_add_frame_to_drizzle_accumulators` (queue_manager.py
  ~5473, def ~15260) ; accumulateurs `drizzle_accumulators` (3 canaux) initialisés
  dans `initialize` (~2723) ; `_decide_finalization_mode` (~325) = source de
  vérité unique ; `_save_final_stack` branch drizzle (~12817-12871) = finalize
  unique depuis les accumulateurs.
- `drizzle_mode` ("Final"/"Incremental") : attribut conservé mais **inerte**
  (logs "SANS EFFET"), GUI conserve encore les radios Final/Incremental.
- Preview drizzle : **absente en M3** (`cumulative_drizzle_data` n'est rempli
  que dans la méthode morte `_process_incremental_drizzle_batch` ~8891 ;
  `refresh_preview` retombe sur `_update_preview_sum_w` qui sort tôt sans memmaps).
- Tests drizzle : 27 passed / 1 failed pré-existant documenté
  (`test_save_final_stack_radec_from_reference_header` — chemin classique RA/Dec,
  hors périmètre M3-D).
- Environnement : `.venv/bin/python -m pytest` (python système sans cv2).

## Milestones

### M3-D-1 — Politique de traitement + grouped incremental + preview + logs + tests synthétiques — ACCEPTÉ
- `drizzle_processing_policy` ∈ {standard, incremental} dérivé de `drizzle_mode`
  (Final→standard, Incremental→incremental), + `drizzle_group_size` (défaut ~50).
- Worker : accumulation par frame identique pour les 2 politiques ; en
  incrémental : compteur de groupe, preview + logs à chaque groupe plein et au
  dernier groupe partiel ; libération explicite.
- `_update_preview_drizzle_accumulator()` : preview = finalize("divide") par
  canal depuis l'accumulateur (copies), stretch/downsample, callback preview ;
  display-only.
- Logs : `DRIZZLE POLICY: STANDARD` / `DRIZZLE POLICY: INCREMENTAL (group_size=N)`,
  `group n/N frames accumulated: K preview updated`. Finalisation identique.
- Tests : invariant standard vs incremental (SCI/WHT/WCS), group_size 2/20/200,
  dernier groupe partiel, aucune pose double/oubliée, preview non-interférente,
  mémoire non linéaire en N, non-régression classic/reproject.
- Ne pas toucher : drizzle_core.py, _decide_finalization_mode, _save_final_stack
  branch drizzle, classic, mosaïque, reproject.



#### Revue M3-D-1 (2026-08-20)
- Diff inspecté : `seestar/queuep/queue_manager.py` (+189/-2) + `tests/test_m3d_policy.py`. `drizzle_core.py`, `_decide_finalization_mode` et le chemin scientifique de `_save_final_stack` restent inchangés.
- Tests reviewer : 49 passed (m3d/drizzle_core/worker/drizzle_finalize/interbatch/queue_manager_reproject), 6 passed (m3d only), 23 passed (incremental_reprojection + reproject_mode_consistency), 13 passed + 1 failed pré-existant (`test_save_final_stack_radec_from_reference_header`).
- Invariance confirmée par tests : Standard == Incremental, group_size 2/20/200, dernier groupe partiel, preview non-interférente, mémoire accumulateur indépendante de N.
- Follow-up mineur pour M3-D-2 : le compteur `_drizzle_frame_count` doit idéalement compter toutes les poses drizzle (standard aussi) afin que la preview manuelle standard affiche un count exact ; ajouter aussi une validation robuste de `drizzle_group_size` côté settings/UI.

### M3-D-2 — GUI + settings (politique, pas deux algorithmes) — IMPLÉMENTÉ (revue en attente)
- Radios : "Standard" / "Large dataset / incremental" (valeurs conservées
  Final/Incremental), nettoyage `_update_drizzle_options_state` (~632),
  special-case preview ~3665, compteur images ~5591, settings.
- Label mode -> "Drizzle processing:". `drizzle_group_size_var` + Spinbox
  "Preview group size:" (enabled seulement Drizzle coché ET Incremental).
  Settings: default 50, update_from_ui/apply_to_ui/validate/save + compat
  anciens JSON sans la clé. Compteur `_drizzle_frame_count` compte désormais
  toutes les poses drizzle (standard aussi), preview auto groupe uniquement en
  incremental. Tests: `tests/test_m3d_settings.py` + 2 tests tick dans
  `tests/test_m3d_policy.py`.



#### Revue M3-D-2 (2026-08-20)
- Diff inspecté : `main_window.py`, `settings.py`, micro-polish `queue_manager.py`, `tests/test_m3d_settings.py` + extension `tests/test_m3d_policy.py`. Valeurs persistées `Final`/`Incremental` conservées ; labels visibles remplacés par `Standard` / `Large dataset / incremental`.
- Défaut reviewer corrigé localement : trailing whitespace dans lignes GUI/settings ajoutées (CRLF mix historique préservé, logique inchangée). `git diff --check` OK après correction.
- Tests reviewer : 29 passed (M3-D/settings/drizzle), py_compile OK ; puis 60 passed (M3-D + drizzle + interbatch + queue_manager_reproject) ; 23 passed séparés (incremental_reprojection + reproject_mode_consistency).
- Invariant UI validé : `drizzle_group_size` est une politique de preview/progression, activée seulement en Large dataset ; scale/WHT/kernel/pixfrac ne sont plus présentés comme dépendants d'un algorithme différent.

### M3-D-3 — Retraite du code historique + étude checkpoint — IMPLÉMENTÉ (revue en attente)
- Audit des références vivantes (drizzle_executor, _wait_drizzle_processes,
  _start_drizzle_process, drizzle_batch_worker, _save_drizzle_input_temp,
  _process_incremental_drizzle_batch, incremental_drizzle_objects/*_sci/*_wht,
  intermediate_drizzle_batch_files, _update_preview_incremental_drizzle) ;
  suppression ou marquage obsolète clair ; doc `docs/M3D_checkpoint_design.md`.

#### Résultat M3-D-3 (2026-08-20)
- **Aucun retrait de code** : le chemin legacy « double-pass » est conservé
  uniquement car `tests/test_save_final_stack.py` appelle directement
  `_process_incremental_drizzle_batch` (weight-override / accumulation), et le
  cycle de vie de `drizzle_executor`/`_wait_drizzle_processes` est encore
  référencé par `seestar/gui/boring_stack.py` (hors périmètre M3-D-3).
- **Marquage « M3-D OBSOLETE LEGACY »** (docstrings/commentaires sans
  ambiguïté) sur : `drizzle_batch_worker`, `_start_drizzle_process`,
  `_wait_drizzle_processes`, `_process_incremental_drizzle_batch`,
  `_update_preview_incremental_drizzle`, `incremental_drizzle_objects`,
  `intermediate_drizzle_batch_files`, `GLOBAL_DRZ_BATCH_VERSION_STRING_ULTRA_DEBUG`.
- `_save_drizzle_input_temp` et `_process_and_save_drizzle_batch` : **déjà
  absents** du `queue_manager.py` vivant (présents seulement dans `build/lib/`,
  artefact obsolète) — aucune action nécessaire.
- `cumulative_drizzle_data`/`_raw` : **conservés** comme artefact
  display-only de la preview M3 (`_update_preview_drizzle_accumulator`),
  commentés « non scientifique, jamais réinjecté ».
- Doc design : `docs/M3D_checkpoint_design.md` (état accumulateur
  `_out_img`/`_out_wht`, WCS/shape/kernel/pixfrac/fillval, manifest/hash,
  dernier index, compteurs, version schéma, format `.npz`/FITS+JSON, risques,
  tests futurs). Aucune implémentation de checkpoint.
- Validation : `git diff --check` OK, `py_compile` OK, suites drizzle/M3-D
  exécutées (voir rapport M3-D-3). Aucun commit, aucun push.



#### Revue M3-D-3 (2026-08-20)
- Audit Coco vérifié par Jarvis : 9 symboles legacy marqués `M3-D OBSOLETE LEGACY` (docstrings/commentaires uniquement, zéro changement de logique). `_process_incremental_drizzle_batch` conservé car réellement appelé par `tests/test_save_final_stack.py` ; `drizzle_executor`/`_wait_drizzle_processes` conservés pour le cycle de vie `boring_stack.py`. `cumulative_drizzle_data` = KEPT-LIVE display-only. `FINALIZATION_MODE_DRIZZLE` unique (grep).
- `docs/M3D_checkpoint_design.md` créé (DESIGN ONLY) : état `_out_img`/`_out_wht` par canal, WCS/shape/kernel/pixfrac/fillval, manifest/hash, dernier index, version schéma, format .npz/FITS+JSON atomic, risques, tests futurs. Gap identifié : `fillval` non conservé sur `DrizzleAccumulator` (à ajouter si checkpoint implémenté).
- Tests reviewer : 29 passed ciblés ; 13 passed + 1 failed pré-existant (radec_from_reference_header) côté legacy. `git diff --check` OK.

#### Revue M3-D-4 (2026-08-20)
- Suite complète en un process : 10 erreurs de collection — reproduites À L'IDENTIQUE sur HEAD bc87eb3 (pollution d'import pré-existante entre fichiers de test ; le repo se valide par lots).
- Validation par lots (process frais par fichier) : 51/57 fichiers verts dans l'arbre M3-D ; les 6 fichiers à échecs (astap_wcs_padding, load_wcs_ignore_missing_simple, preserve_linear_output, progress_callback, reproject_utils, save_final_stack) échouent STRICTEMENT À L'IDENTIQUE sur HEAD → tous pré-existants, zéro régression M3-D.

### M3-D-4 — Validation finale + witness optionnel — DONE (witness : porte Tristan)
- Suite complète pytest, non-régression, comparaison Standard vs Incremental sur
  mêmes poses (witness M16 court dans /home/tristan/M16/quick/, non répété).

### Micro-amendement — Frontière boring_stack vs politique incremental (2026-08-20)

Statut : **IMPLÉMENTÉ** (en attente de revue).

Objectif : rendre EXPLICITE que `seestar/gui/boring_stack.py` est strictement
le chemin classique mono-lot (SUM/W memmaps) et qu'il ne sélectionne ni
n'implémente jamais la politique Drizzle Incremental (celle-ci reste
exclusivement dans `queue_manager.py`).

Preuves vérifiées (code + tests) :

- `boring_stack.py` appelle `start_processing(... use_drizzle=False ...)`
  (~L882) et ne transmet PAS `drizzle_mode` → `drizzle_active_session`
  reste `False` (porte de session `use_drizzle or is_mosaic_run`), donc
  `drizzle_accumulators` n'est jamais créé (`initialize` est gate sur la
  session). Aucune politique incremental, aucun accumulateur/lifecycle legacy.
- Aucun preview par groupe ni `drizzle_group_size` dans `boring_stack.py`
  (grep : 0 occurrence des symboles `_drizzle_group_tick`, `drizzle_group_size`,
  `_process_incremental_drizzle_batch`, `_start_drizzle_process`,
  `_update_preview_drizzle_accumulator`).
- Seul contact legacy : `_cleanup_stacker` (~L82) appelle
  `stacker._wait_drizzle_processes()` — no-op M3 (marqué OBSOLETE dans
  M3-D-3), conservé pour le cycle de vie shutdown des executors, et draine
  `drizzle_executor`/`quality_executor`. Docstring élargie (commentaire
  uniquement, zéro changement de logique) pour documenter cette frontière.
- Mono-lot / OOM borné : memmaps classiques `cumulative_sum_memmap` /
  `cumulative_wht_memmap` alloués avec la shape fixe de la grille de référence
  `(H, W, C)` / `(H, W)` (queue_manager ~L2814-2829) — indépendante du nombre
  de poses N → O(HxW), pas O(N) ; align_on_disk auto activé > 50 images
  (~L734-741), batch_size=1 → fichiers alignés sur disque.

Fichiers :

- `seestar/gui/boring_stack.py` : docstring de `_cleanup_stacker` (frontière,
  `_wait_drizzle_processes` = legacy no-op M3). Aucun changement de logique.
- `tests/test_boring_drizzle_boundary.py` : 10 tests (isolation politique,
  inertie tick/flush/preview, `_cleanup_stacker` sans effet incremental,
  audit source, mono-lot/OOM shape fixe).
- `docs/M3D_MISSION.md` : cette section.

Validation : `test_boring_drizzle_boundary.py` (10 passed),
`test_m3d_policy.py` + `test_m3d_settings.py` (17 passed),
`test_boring_thread.py` + `test_boring_thread_regex.py` (2 passed),
`git diff --check` OK, `py_compile` OK. Aucun commit, aucun push.

## Règles durables
- Aucun push. Commits locaux seulement, après validation du revieweur.
- Aucun run long répété ; datasets synthétiques petits (16-32 px) pour les tests.
- Preuves par tests + logs ; jamais de PASS déclaré avant runs réels.
