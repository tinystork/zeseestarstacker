# ZSSS W-1 — Audit de flux : cohérence des modes de traitement

Repo : `zeseestarstacker` — branche `beta` — HEAD `f7dd0b2` (M2d).
Fichier principal : `seestar/queuep/queue_manager.py`.

## 1. Objectif

Reconstituer le flux réel `settings → start_processing → initialize → worker →
finalisation` **sans présupposer la root cause**, puis documenter le scénario
exact qui produit le message witness :

> `Erreur obtention donnees brutes finales: Accumulateurs memmap SUM/WHT non
> disponibles pour stacking classique.`

(levé dans `_save_final_stack`, branche `else` SUM/W classique).

## 2. États et transitions par phase

### 2.1 Valeurs initiales (constructeur `__init__`)

| État | Valeur | Ligne (HEAD) |
|---|---|---|
| `drizzle_active_session` | `False` | 1930 |
| `is_mosaic_run` | `False` | 1929 |
| `reproject_between_batches` | `False` | 1976 |
| `reproject_coadd_final` | `False` | 1977 |
| `freeze_reference_wcs` | `False` | 2129 |
| `cumulative_sum_memmap` / `cumulative_wht_memmap` | `None` | 2071-2072 |
| `drizzle_accumulators` | `None` | 2149 |
| `finalization_mode` | `None` (ajout W-1) | — |

### 2.2 Lecture des Settings (fin du constructeur, ~2156-2176)

```python
self.reproject_between_batches = bool(getattr(settings, "reproject_between_batches", False))
self.freeze_reference_wcs = self.reproject_between_batches
self.reproject_coadd_final = bool(
    getattr(settings, "reproject_coadd_final", False)
    or getattr(settings, "stack_final_combine", "") == "reproject_coadd"
)
```

→ Un settings « hérité » peut porter `reproject_between_batches=True` **ou**
`reproject_coadd_final=True` (ou `stack_final_combine == "reproject_coadd"`)
**indépendamment** de `use_drizzle`. C'est la source des combinaisons de flags
incohérentes.

### 2.3 `start_processing` (réapplication des flags)

- `self.is_mosaic_run = is_mosaic_run` (14236)
- `self.drizzle_active_session = use_drizzle or self.is_mosaic_run` (14237)
- `reproject_between_batches` / `reproject_coadd_final` sont réappliqués depuis
  les arguments **seulement s'ils ne sont pas `None`** (14325 / 14334).
- `batch_size == 0` force (14440-14490) : `reproject_between_batches=False`,
  `freeze_reference_wcs=True`, et `reproject_coadd_final=True` si non déjà fixé.

### 2.4 `initialize` (initialisation de l'accumulation)

```python
is_drizzle_standard_mode = self.drizzle_active_session and not self.is_mosaic_run
```

- `True`  → création des `drizzle_accumulators` (1 par canal) **et**
  `cumulative_sum_memmap = cumulative_wht_memmap = None`.
- `False` → création des memmaps SUM/WHT.

Point critique (pré-fix) : la décision d'accumulation ne regarde **que**
`drizzle_active_session` / `is_mosaic_run`. Elle **ignore** les flags
`reproject_*`. Donc `drizzle_active_session=True + reproject_coadd_final=True`
→ **accumulation Drizzle** (accumulateurs créés, memmaps `None`).

### 2.5 Worker — finalisation (`_worker`, ~5750-6030)

Branches de finalisation :

| Branche | Condition | Appel `_save_final_stack` | Données consommées |
|---|---|---|---|
| Mosaïque | `is_mosaic_run` | `_mosaic_reproject`, sci/wht | SCI/WHT |
| Drizzle | `drizzle_active_session` | `_drizzle_final` (sans sci/wht) | `drizzle_accumulators` |
| Reproject entre lots | `reproject_between_batches` | `_classic_reproject` (sans sci/wht) | memmaps SUM/W |
| Reproject&Coadd | `reproject_coadd_final` | `_classic_reproject`, sci/wht | SCI/WHT (`current_stack`/`current_coverage`) |
| Classique SUM/W | sinon | `_classic_sumw` (sans sci/wht) | memmaps SUM/W |

### 2.6 `_save_final_stack` — détection du mode (AVANT fix)

Quatre booléens inférés **à partir des flags ET de la présence des données** :

```python
is_reproject_mosaic_mode       = suffix == "_mosaic_reproject" and sci/wht is not None
is_drizzle_standard_from_accumulators = drizzle_active_session and not is_mosaic_run
                                 and not reproject_between_batches
                                 and not reproject_coadd_final
                                 and sci is None and accumulators is not None
                                 and not is_reproject_mosaic_mode
is_classic_reproject_mode      = (reproject_between_batches or reproject_coadd_final)
                                 and sci is not None and wht is not None
is_classic_stacking_mode       = memmaps is not None and not (les trois autres)
else:  # FALLBACK
    if not drizzle_active_session and not is_mosaic_run:
        is_classic_stacking_mode = True   # ← fallback arbitraire
```

## 3. Scénario exact du message memmap (witness)

Valeurs réelles des flags dans le scénario :

- `use_drizzle = True` → `drizzle_active_session = True`
- settings hérité : `reproject_coadd_final = True` (ou `reproject_between_batches = True`)
- `is_mosaic_run = False`

Déroulé :

1. `initialize` : `is_drizzle_standard_mode=True` → **accumulateurs Drizzle créés**,
   `cumulative_sum_memmap = cumulative_wht_memmap = None`.
2. `_worker` : `elif self.drizzle_active_session:` → `_save_final_stack("_drizzle_final")`
   **sans** sci/wht.
3. `_save_final_stack` (pré-fix) :
   - `is_drizzle_standard_from_accumulators` → **False** car `reproject_coadd_final=True`.
   - `is_classic_reproject_mode` → **False** car sci/wht absent.
   - `is_classic_stacking_mode` → **False** car memmaps `None`.
   - `else` (fallback) : `not drizzle_active_session` est **False** (drizzle actif),
     donc `is_classic_stacking_mode` reste `False`, mode log `"Unknown"`.
4. Le bloc `try` n'atteint aucune branche if/elif et tombe dans le `else` SUM/W :
   `if cumulative_sum_memmap is None or cumulative_wht_memmap is None: raise
   ValueError("Accumulateurs memmap SUM/WHT non disponibles …")`.

→ **Une stratégie d'accumulation (Drizzle) est initialisée, puis une AUTRE
(SUM/W classique) est finalisée.** C'est exactement l'incohérence signalée.

Variante équivalente : `drizzle_active_session=True + reproject_between_batches=True`
produit le même chemin (le flag `reproject_between_batches` bloque aussi
`is_drizzle_standard_from_accumulators`).

## 4. Correction appliquée

### 4.1 Modèle de mode retenu

Un mode = une stratégie d'accumulation/finalisation cohérente. Le mode est
**décidé une seule fois**, à l'initialisation de l'accumulation (`initialize`),
via `_decide_finalization_mode`, puis **transmis explicitement** à
`_save_final_stack` (attribut d'instance `self.finalization_mode` + paramètre
optionnel). `_save_final_stack` **consomme** ce mode ; plus aucun fallback
arbitraire.

Ordre de priorité (source de vérité unique) :

| Priorité | Flag | Mode |
|---|---|---|
| 1 | `is_mosaic_run` | `mosaic` |
| 2 | `drizzle_active_session` | `drizzle` (précède tout flag reproject) |
| 3 | `reproject_between_batches` | `classic_sumw` (lots reprojetés sommés en memmaps) |
| 4 | `reproject_coadd_final` | `reproject_coadd` (SCI/WHT produit explicitement) |
| 5 | défaut | `classic_sumw` |

### 4.2 Contrat de données par mode (jamais de fallback)

- `mosaic` / `reproject_coadd` : **exigent** `drizzle_final_sci_data` et
  `drizzle_final_wht_data` ; sinon `ValueError` clair.
- `drizzle` : lit les `drizzle_accumulators`.
- `classic_sumw` : lit les memmaps SUM/WHT.

### 4.3 Garde en amont (`_check_finalization_ready`)

Appelée dans le worker **avant** `_save_final_stack` :

- `drizzle` sans accumulateurs ou à poids tous nuls → échec propre
  « Drizzle: aucune image accumulée (poids tous nuls) ».
- `classic_sumw` sans memmaps → échec propre
  « Stacking classique: accumulateurs memmap SUM/WHT non disponibles ».
- `classic_sumw` avec 0 image accumulée → échec propre
  « Stacking classique: aucune image accumulée ».

Aucune sélection arbitraire d'un autre mode.

## 5. Diff exact (fichiers modifiés)

- `seestar/queuep/queue_manager.py`
  - Ajout module-level : constantes `FINALIZATION_MODE_*`, `_FINALIZATION_MODES`,
    `_decide_finalization_mode`, `_drizzle_accumulators_have_data`.
  - `__init__` : `self.finalization_mode = None`.
  - `initialize` : `self.finalization_mode = _decide_finalization_mode(self)` + log.
  - `_save_final_stack` : signature + résolution de mode explicite + suppression
    du `else` fallback + validation du contrat SCI/WHT + log du mode finalisé.
  - `_check_finalization_ready` : nouvelle méthode de garde amont.
  - `_worker` : garde amont sur les branches Drizzle / Reproject entre lots /
    SUM/W, et sur la sauvegarde partielle Drizzle.
- `tests/test_save_final_stack.py`
  - Alignement des tests sur le contrat explicite (mode mosaic / reproject_coadd /
    accumulateurs Drizzle).
- `tests/test_reproject_mode_consistency.py` (nouveau) : 22 tests.

Hors périmètre non touchés : `astrometry_solver.py`, `zesolver_adapter.py`,
`local_solver_gui.py`, noyau `drizzle_core.py` (aucune modification), chemins
non concernés, sémantique de `batch_size`, pipeline de solve.

## 6. Justification du modèle de mode

Le bug vient de l'inférence de mode **au moment de la finalisation** à partir
d'une conjonction de flags + de la présence des données, avec un `else` qui
déclarait arbitrairement « classic SUM/W » quand rien ne correspondait. Cette
inférence autorisait un état incohérent (accumulation Drizzle + flags
reproject hérités) à glisser jusqu'au `else` SUM/W.

Le modèle retenu sépare **décision** (une fois, à l'initialisation de
l'accumulation) et **consommation** (finalisation), en une source de vérité
unique `_decide_finalization_mode`. La priorité explicite « Drizzle précède
tout flag reproject » rend impossible la finalisation d'une accumulation
Drizzle comme SUM/W, quelle que soit la combinaison de flags hérités. La garde
amont garantit qu'un mode n'est finalisé que si son accumulation est
effectivement disponible (échec propre sinon), et le contrat de données par
mode élimine tout basculement silencieux vers un autre chemin.

## 7. Résultats de validation

Commande : `.venv/bin/python -m pytest <suite> -q`

- `tests/test_reproject_mode_consistency.py` : **22 passed**.
- `tests/test_save_final_stack.py` : 10 passed, 1 failed
  (`test_save_final_stack_radec_from_reference_header`, **pré-existant** — voir §8).
- Suites drizzle (`test_drizzle_core`, `test_drizzle_finalize`,
  `test_drizzle_integration`, `test_drizzle_integration_qm`,
  `test_drizzle_integrator`, `test_drizzle_legacy_behavior`,
  `test_drizzle_preview`) : **toutes vertes**.
- Suites reproject/queue_manager (`test_queue_manager_reproject`,
  `test_reproject_utils`, `test_reproject_zm_wcs_fix`,
  `test_incremental_reprojection`, `test_worker_incremental_drizzle`) :
  39 passed, **13 failed — tous pré-existants à HEAD** (vérifiés par `git stash`,
  ensemble identique) — voir §8.

## 8. Échecs pré-existants (documentés et vérifiés à HEAD)

1. `test_save_final_stack_radec_from_reference_header` : le double léger
   (`Dummy`) n'a pas d'attribut `logger` ; la branche SUM/W appelle
   `self.logger.info(...)` → `AttributeError` → `final_stacked_path=None`.
   Pré-existant, hors périmètre (harness de test, pas le flux de mode).
2. `test_reproject_utils.py` : `NameError: ReprojectCoaddResult` /
   `_estimate_mem_gb` non définis et `ImportError: ReprojectCoaddResult`
   dans `seestar/enhancement/reproject_utils.py`. Pré-existants.
3. `test_queue_manager_reproject.py` : 6 tests (`test_reproject_classic_batches_*`)
   échouent pour les mêmes raisons `reproject_utils`. Pré-existants.
4. `test_preserve_linear_output.py` : échec de collecte
   `ModuleNotFoundError: No module named 'seestar.queuep.autotuner'`
   (chargeur custom du module). Pré-existant.

Aucun de ces échecs n'est causé par la correction W-1 ; l'ensemble des échecs
est identique à HEAD (`git stash` vérifié).
