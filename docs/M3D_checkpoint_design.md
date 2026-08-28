# M3-D — Design de checkpoint / reprise (D1 writer implémenté, reprise NON implémentée)

Statut : **D1 (writer) IMPLÉMENTÉ** — écriture atomique du checkpoint natif
Drizzle (RSM2-D1).  La **reprise** (reader / restore / activation de Resume)
reste **NON implémentée** et n'est pas autorisée à démarrer automatiquement
(voir §7).  Ce document fige l'état mathématique à préserver, le protocole
d'écriture copy-on-write effectivement livré en D1, et les contraintes que
devra respecter une éventuelle mission de reprise **sans changer la science**
du drizzle M3.

## 1. Objectif

Permettre d'interrompre et de reprendre un traitement *Large dataset /
incremental* sans modifier le résultat scientifique.

Invariant (repris de `docs/M3D_MISSION.md`) :

> Pour les mêmes poses, transformations, poids et paramètres Drizzle, un run
> continu et un run checkpoint/reprise doivent produire le **même SCI**, le
> **même WHT** et le **même WCS final**, à une tolérance numérique définie.

Le checkpoint est un **état mathématique de l'accumulateur**, et non une
mémoire de batching : le group (`drizzle_group_size`) reste une pure politique
de ressources/progression, jamais une unité de combinaison.

## 2. Ce que N'EST PAS un checkpoint

- ❌ Une image drizzle intermédiaire (SCI/WHT normalisée) destinée à être
  **re-drizzlée** — c'est précisément l'ancien chemin « double-pass » invalidé
  (voir `tests/test_drizzle_legacy_behavior.py`) : il dépend de la taille de
  lot et replie le flux des bords.
- ❌ Un aperçu (`cumulative_drizzle_data`) — artefact **display-only**, jamais
  réinjecté dans le calcul.
- ❌ Un dump des poses déjà lues — les poses originales restent la source ;
  seul leur **index de progression** est mémorisé.

## 3. État à sérialiser

### 3.1 Par canal — état de `DrizzleAccumulator` (source de vérité)

`seestar/core/drizzle_core.py`, classe `DrizzleAccumulator`. Rappel sémantique
(`drizzle` 2.2.0) : `out_img` contient la **moyenne pondérée**, `out_wht` le
**poids total** ; le flux pondéré vaut `out_img * out_wht`.

Pour chaque canal `c ∈ {0,1,2}` :

| Champ | Type | Rôle |
|---|---|---|
| `_out_img[c]` | `float32 (H,W)` | état interne drizzle = **weighted mean** (`out_img`) |
| `_out_wht[c]` | `float32 (H,W)` | **poids total** accumulé (`out_wht`) |

Sérialiser **ces deux tableaux bruts** (pas `sci`/`wht` normalisé) est la
condition de reprise exacte : `Drizzle.add_image` mute `out_img`/`out_wht` en
place, donc restaurer ces deux tableaux et y ré-attacher un `Drizzle` reconstruit
donne une continuation identique (ordre d'accumulation préservé).

**Contrat de reconstruction vérifié** (implémenté par
`DrizzleAccumulator.from_native_state`, prouvé par
`tests/test_drizzle_resume_continuation.py`). L'état de reprise d'un canal est
exactement :

* les deux tableaux natifs `float32 (H,W)` **finis** `out_img` / `out_wht`
  (le WHT peut être **signé** pour les noyaux Lanczos) ;
* le `total_exptime` accumulé (somme des `exptime` acceptés), **fini et non
  négatif** ;
* les paramètres runtime effectifs `kernel` / `pixfrac` / `fillval`.

La reconstruction passe par la méthode dédiée (jamais par une ré-affectation
directe de l'objet interne) :

```python
acc = DrizzleAccumulator.from_native_state(
    out_shape_hw,
    restored_out_img,          # float32, fini
    restored_out_wht,          # float32, fini (signé OK pour Lanczos)
    kernel=kernel,             # valeur runtime effective
    pixfrac=pixfrac,           # valeur runtime effective (1.0 pour Lanczos)
    fillval=fillval,           # valeur runtime effective
    total_exptime=total_exptime,  # fini, >= 0
)
```

Deux contraintes `drizzle` 2.2.0 rendent ce chemin nécessaire et suffisant
(vérifiées contre la bibliothèque installée) :

* un `out_wht` pré-rempli avec `exptime == 0` lève une erreur (« Exposure time
  cannot be 0 when context and/or weight arrays are non-zero ») → le
  `total_exptime` accumulé doit être restauré avec les tableaux ;
* un `out_wht` pré-rempli sans `out_ctx` assorti lève une erreur (« Pixels with
  non-zero context values must have positive weights and vice-versa ») → le
  bitmap de contexte (bookkeeping uniquement, jamais partie de l'invariant
  science) n'est pas persisté et la reconstruction désactive son suivi
  (`disable_ctx=True`), ce qui ne change **aucune** science (`out_img`/`out_wht`
  restent bit-identiques).

`DrizzleAccumulator` **retient** `fillval` (`self.fillval`), ainsi que
`kernel` / `pixfrac` ; ces trois paramètres sont donc persistés puis restaurés
comme paramètres runtime (via `from_native_state`), pas comme attributs internes
devinés.

### 3.2 Géométrie et paramètres Drizzle (partagés par les 3 canaux)

| Champ | Rôle |
|---|---|
| `drizzle_output_wcs` | WCS de sortie (CRVAL/CTYPE/CDELT/CRPIX) — identique pour les 3 canaux |
| `out_shape_hw` / `drizzle_output_shape_hw` | `(H, W)` de la grille de sortie |
| `kernel` | `"square"` (défaut core, non exposé en M3) |
| `pixfrac` | `1.0` (défaut core, non exposé en M3) |
| `fillval` | `"0.0"` |
| `reference_wcs_object` | WCS de référence (nécessaire si `tf` est utilisé pour calculer le pixmap) |

### 3.3 Manifest source et progression

| Champ | Rôle |
|---|---|
| `source_manifest` | liste ordonnée des poses : chemin, taille, `hash` (sha256) / `mtime`, `EXPTIME`, ordre |
| `last_processed_index` | **dernier index de pose traitée** (0-based, inclusif). Le run reprend à `index+1`. |
| `drizzle_processing_policy` | `standard` / `incremental` (politique, pas science) |
| `drizzle_group_size` | taille de groupe (progression/preview uniquement) |
| `_drizzle_frame_count` | compteur de poses accumulées |
| `_drizzle_group_index` | index de groupe courant (preview/log) |
| `images_in_cumulative_stack` / `total_batches_estimated` | compteurs progression/UI |
| `finalization_mode` | `FINALIZATION_MODE_DRIZZLE` (ré-affirmé à la reprise, jamais ré-dérivé) |

> Manifest hash obligatoire : si le contenu des poses a changé entre le
> checkpoint et la reprise (fichier déplacé/modifié), la reprise doit **refuser**
> plutôt que produire un résultat silencieusement différent.

### 3.4 Version de schéma

| Champ | Rôle |
|---|---|
| `schema_version` | entier/semver ; refus propre d'un checkpoint de version inconnue |
| `drizzle_lib_version` | version de `drizzle` (et `numpy`) utilisée à l'écriture |
| `producer` | `zeseestarstacker` + version |

## 4. Contraintes de reprise

1. **Même science, même tolérance.** La reprise ré-injecte uniquement
   `_out_img`/`_out_wht` dans des accumulateurs reconstruits ; elle ne re-drizzle
   jamais une image intermédiaire.
2. **Ordre des poses préservé.** Le run reprend strictement après
   `last_processed_index` dans `source_manifest` ; l'ordre d'accumulation est
   conservé (l'addition flottante n'est pas commutative, la tolérance doit en
   tenir compte).
3. **Tolérance définie.** Comparer `finalize("divide")` continu vs reprise avec
   `np.allclose(..., rtol=1e-6, atol=1e-7)` + comparaison WCS (CDELT/CRPIX à
   `1e-9` près) et WHT (`rtol=1e-6`). Documenter que la reprise est bit-identique
   pour les poses suivantes mais que l'arrondi dépend de l'ordre.
4. **Finalisation unique inchangée.** La reprise ne modifie ni
   `_decide_finalization_mode` ni le chemin drizzle de `_save_final_stack`
   (`FINALIZATION_MODE_DRIZZLE`, `finalize` unique depuis `drizzle_accumulators`).
5. **Interruption au milieu d'une pose.** Le checkpoint n'est écrit qu'**entre**
   deux poses (après le `add` complet d'une pose), jamais pendant un `add_image`
   partiel (3 canaux = 3 appels `add` ; un checkpoint entre canaux laisserait un
   état incohérent). Écrire donc à la frontière de pose uniquement.

## 5. Format recommandé

### 5.1 Format effectivement livré (D1)

Implémenté par `seestar/core/drizzle_checkpoint.py`
(`DrizzleCheckpointWriter`), intégré dans `queue_manager` aux frontières de pose
acceptées.  Namespace **dédié** — jamais les artefacts classiques
`memmap_accumulators/resume_manifest.json` (qui restent réservés au SUM/W
classique et ne sont ni surchargés ni affaiblis).

- **Conteneur :** six fichiers `.npy` float32 (pas d'archive `.npz`, pas de
  FITS) : un `out_img` et un `out_wht` par canal, sous des noms **uniques par
  génération**.  Le WCS n'est pas dans les tableaux (il est dans le manifest).
- **Métadonnées :** `checkpoint.json` (manifest déterministe JSON,
  `sort_keys=True`) : géométrie, WCS sérialisé, paramètres effectifs, compteurs,
  empreinte scientifique, `scientific_config`/`run_config_digest`, ledger.
- **Checksum :** SHA-256 **sur les octets exacts de chaque fichier `.npy`**
  (et taille exacte), vérifiables à la reprise avant toute réinjection.
- **Config canonique :** `run_config.cfg` (schéma v2, atomique) écrit avant le
  manifest ; `run_config_digest` + `scientific_config` + empreinte
  `run_contract.drizzle_fingerprint` sont embarqués dans le manifest.
- **Compat versionnée :** `schema_version` / `mode` (`drizzle_native_v1`)
  refusés s'ils sont inconnus à la reprise ; `drizzle_lib_version` et
  `numpy_version` consignés pour diagnostiquer une dérive de bibliothèque.

Exemple de layout effectif (génération `00000003`) :

```
{output_folder}/.m3d_checkpoint/
    checkpoint.json                        # SEUL point de commit
    gen-00000003-ch0-out_img.npy
    gen-00000003-ch0-out_wht.npy
    gen-00000003-ch1-out_img.npy
    gen-00000003-ch1-out_wht.npy
    gen-00000003-ch2-out_img.npy
    gen-00000003-ch2-out_wht.npy
{output_folder}/run_config.cfg             # config canonique (stable)
```

### 5.2 Protocole copy-on-write / commit (manifest-last)

1. Valider tout l'état **avant** toute écriture (échec fermé, jamais de
   génération partielle/mixte) : 3 accumulateurs de même shape/config/exptime,
   buffers float32 finis, compteurs finis, liaison de session présente, ledger
   sans doublon ni source non identifiable, config canonique bien formée.
2. Écrire les six artefacts natifs sous des noms uniques de génération, via
   fichier temporaire du même répertoire + `os.replace` (fsync avant replace).
3. Calculer SHA-256 + taille exacte de chaque artefact final.
4. Écrire `run_config.cfg` (atomique, contenu stable) **avant** le manifest.
5. Écrire `checkpoint.json.tmp`, fsync/close, puis `os.replace` vers
   `checkpoint.json` **en dernier** : `checkpoint.json` est le seul point de
   commit.

Invariants de sûreté :

- Un fichier référencé par le manifest actuellement commité n'est **jamais**
  écrasé avant que le nouveau manifest ne soit commité (les générations sont
  uniques par nom).
- En cas d'échec avant le `os.replace` du manifest : le manifest précédent et
  tous ses fichiers restent **byte-identiques et utilisables** ; les
  fichiers temporaires/générations non commitées de la tentative sont nettoyés
  au mieux (jamais les fichiers non liés).
- Après un commit réussi, les générations précédentes peuvent être
  garbage-collectées au mieux **uniquement** depuis l'allowlist explicite
  `gen-*-ch[0-2]-out_(img|wht).npy` — jamais de suppression large du
  répertoire, jamais la génération courante.

### 5.3 Cadence et limitations

- Cadence : un snapshot **tous les `drizzle_group_size` poses acceptées**,
  indépendamment de la politique de preview Standard/Incremental ; pas de
  snapshot par pose (I/O borné), pas de checkpoint entre les 3 adds de canal,
  pas de checkpoint sur un add échoué.
- **Force finale** : un snapshot propre du groupe partiel traînant est forcé
  sur Stop ordonné et sur finalisation réussie (via les hooks de fin sûrs
  existants) ; idempotent si aucun état n'a changé depuis le dernier commit.
- Premier commit = au moins une pose acceptée (jamais de checkpoint vide).
- La **reprise** (lecture/restauration/activation de Resume) n'est **pas**
  implémentée en D1 : le writer ne lit jamais, ne finalise jamais, et
  `_build_startup_refusal`/`_validate_resume_headless`/Qt refusent toujours le
  Resume Drizzle comme aujourd'hui.

## 6. Risques techniques

1. **État interne de `drizzle.resample.Drizzle`.** `Drizzle` maintient
   `out_img`/`out_wht` (et possiblement des buffers annexes selon la version).
   Ne pas chercher à sérialiser l'objet `Drizzle` lui-même (non garanti
   picklable/stable) ; reconstruire le wrapper autour des arrays restaurés et
   **valider par un test aller-retour** (cf. §7).
2. **Versions drizzle/lib.** Un changement de version de `drizzle`/`numpy` peut
   modifier l'arrondi de `add_image`. Stocker `drizzle_lib_version` et comparer ;
   en cas de mismatch, autoriser avec WARN explicite ou refuser selon la
   politique choisie.
3. **Ordre des poses.** La reprise DOIT respecter `source_manifest` ; un tri
   différent (ex. re-scan du dossier) changerait le résultat. Le manifest est la
   seule autorité sur l'ordre.
4. **Fichiers déplacés/renommés.** Détecté par hash ; refus avec message clair,
   jamais de fallback silencieux sur un autre fichier.
5. **Poids / exptime.** `EXPTIME` est lu par pose dans
   `_add_frame_to_drizzle_accumulators` (défaut `1.0`, `in_units="counts"`,
   `wht_scale=expscale`). Le manifest doit stocker l'`EXPTIME` de chaque pose
   pour re-vérifier à la reprise.
6. **Interruption au milieu d'une pose.** Écrire le checkpoint uniquement entre
   poses (frontière de pose) ; voir §4.5.
7. **`fillval` / paramètres runtime.** `fillval` est retenu par l'accumulateur
   (`self.fillval`) ; le checkpoint doit le persister avec `kernel` / `pixfrac`
   (paramètres runtime effectifs) et les restaurer via `from_native_state`
   (voir §3.1).

## 7. Tests

### 7.1 Exécutés en D1 (writer)

`tests/test_drizzle_checkpoint_writer.py` couvre :

- **A** — roundtrip d'inspection du writer (sans reader de production) :
  parse du manifest, vérification des six descripteurs (fichier, dtype, shape,
  taille, SHA-256), `np.load` et comparaison **bit-exacte** aux snapshots
  possédés, y compris WHT Lanczos **signé**.
- **B** — atomicité de génération / injection de panne à chaque étape matérielle
  (écriture temp / replace d'array, écriture de la config, écriture temp /
  replace du manifest) : l'ancien `checkpoint.json` et chaque ancien artefact
  restent inchangés ; aucun manifest ne référence une génération partielle/mixte.
- **C** — génération N→N+1 réussie : bascule atomique vers les seuls
  descripteurs N+1, génération courante jamais garbage-collectée.
- **D** — témoin d'ordre aux frontières sûres : pas d'écriture après 0 pose ou
  add échoué, pas de checkpoint entre canaux, cadence à `group_size`, force
  finale sur Stop/succès, idempotence, ledger/compteurs cohérents, échec = abort
  avant le déplacement de la source.
- **E** — échec fermé (mismatch config/empreinte, WCS/liaison de session absent,
  source dupliquée/non identifiable, buffers/compteurs non finis) sans nouveau
  manifest commité.

### 7.2 Recommandés pour la mission de reprise (D2, non exécutés)

- **Continuous vs checkpoint/resume** : même `source_manifest`, même WCS, même
  SCI/WHT à la tolérance définie (run continu complet vs run coupé à `k` poses
  puis repris).
- **Interruption en frontière de groupe** : couper exactement à
  `_drizzle_frame_count == k * drizzle_group_size` ; vérifier la reprise.
- **Interruption au milieu d'un groupe** : couper à `k*g + m` (`0 < m < g`) ;
  vérifier la reprise (le dernier groupe partiel est flushé à la finalisation,
  pas au checkpoint).
- **Manifest mismatch** : modifier/renommer une pose après checkpoint → la
  reprise doit échouer proprement.
- **Version mismatch** : `schema_version` inconnue → refus propre.
- **Partial-group preview non scientifique** : après reprise, l'aperçu du groupe
  partiel est dérivé de l'accumulateur (display-only) et n'affecte pas le SCI
  final.

## 8. Inventaire de la mission de reprise (D2) — NON implémentée

La fin de D1 **ne complète pas** RSM2 et **n'autorise pas** le démarrage
automatique de D2.  La reprise devra (au minimum) :

- lire/valider `checkpoint.json` (schema/mode/empreinte/config/WCS/ledger) et
  vérifier SHA-256/taille de chaque artefact avant réinjection ;
- reconstruire les trois accumulateurs via `DrizzleAccumulator.from_native_state`
  (ordre des poses préservé, `total_exptime` restauré) ;
- revalider la liaison de session (racines d'entrée / référence / plan) et le
  ledger (préfixe ordonné du plan), refuser tout fichier modifié/renommé ;
- réaffirmer `FINALIZATION_MODE_DRIZZLE` et reprendre strictement après
  `last_processed_index` ;
- activer le chemin Resume Drizzle dans `_build_startup_refusal` /
  `_validate_resume_headless` / `_validate_and_open_resume` / Qt readiness —
  chemins **volontairement non touchés** par D1.
