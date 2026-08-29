# M3-D — Design de checkpoint / reprise (D1 writer + D2A reader + D2B1 continuation-writer + D2B2A source-resolution seam ; reprise lifecycle NON implémentée)

Statut : **D1 (writer) IMPLÉMENTÉ** — écriture atomique du checkpoint natif
Drizzle (RSM2-D1) — et **D2A (reader) IMPLÉMENTÉ** — lecteur/validateur
*lecture seule* `read_drizzle_checkpoint` qui reconstruit les trois
accumulateurs après validation complète (RSM2-D2A).  **D2B1 (continuation-writer
seam) IMPLÉMENTÉ** — la factory
`DrizzleCheckpointWriter.from_validated_result` ré-arme le writer atomique à la
génération ``N+1`` **uniquement** depuis un `DrizzleCheckpointResult` déjà
validé, en **relisant intégralement** le checkpoint depuis le disque et en
retournant un objet de ré-arm dédié `DrizzleContinuation` (RSM2-D2B1).
**D2B2A (source-resolution seam) IMPLÉMENTÉ** — le lecteur accepte un
`resolver` explicite/opt-in (par défaut strict D2A) et livre la politique
immutable `SafeStackedSourceResolver` (original-ou-`stacked/` déterministe),
portée comme provenance `resolution_policy` et ré-appliquée par la factory de
continuation (RSM2-D2B2A).
L'**activation de Resume** (wiring lifecycle D2B dans `queue_manager` / GUI /
startup) reste **NON implémentée** et n'est pas autorisée à démarrer
automatiquement (voir §7/§8).
Ce document fige l'état mathématique à préserver, le protocole d'écriture
copy-on-write effectivement livré en D1, le contrat du lecteur D2A, et les
contraintes que devra respecter l'activation de reprise **sans changer la
science** du drizzle M3.

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

0. **Refus de redémarrage** (D1 write-only, Reprise désactivée) : un writer
   fraîchement construit refuse **avant toute écriture** dès que
   `<output>/.m3d_checkpoint` est non vide (manifest, artefact de génération
   allowlisté, temp de manifest ou temp de writer).  Un répertoire existant
   vide est autorisé.  Ce refus est appliqué à la construction **et**
   défensivement au premier commit, et préserve chaque octet préexistant.
1. Valider tout l'état **avant** toute écriture (échec fermé, jamais de
   génération partielle/mixte) : 3 accumulateurs de même shape/config/exptime,
   buffers float32 finis, compteurs finis et cohérents (`exposure_min <=
   exposure_max`, `exposure_unknown_count <= frame_count`), liaison de session
   présente, identités de plan/ledger strictes (entiers non-bool, sans
   doublon), config canonique bien formée, et **paramètres de dépôt runtime**
   (`kernel` / `pixfrac` / `fillval` des accumulateurs) **en accord avec les
   champs scientifiques canoniques** (`drizzle_kernel_effective` /
   `drizzle_pixfrac_effective` / `drizzle_fillval`) — préflight writer
   (durcissement de validation, pas de refonte du protocole) qui empêche D1 de
   publier des paramètres runtime en désaccord avec `run_config.cfg`.
2. **Invariant manifest auto-cohérent** : `len(completed_sources) ==
   frame_count`, `completed_sources == plan.sources[:frame_count]` exactement
   (préfixe ordonné) et `frame_count <= len(plan.sources)` ;
   `stacked_batches_count == frame_count` (incrémenté une fois par pose
   acceptée dans le runtime Drizzle courant).
3. Écrire les six artefacts natifs sous des noms uniques de génération,
   **réclamés exclusivement** (`O_CREAT | O_EXCL`, jamais `os.replace` sur un
   chemin préexistant), écrits en place + fsync.
4. Calculer SHA-256 + taille exacte de chaque artefact final ; fsync du
   répertoire de checkpoint après les créations.
5. Écrire `run_config.cfg` (atomique, contenu stable) **avant** le manifest ;
   fsync du répertoire de sortie après sa publication.
6. Écrire un **temp de manifest possédé et unique par tentative**
   `checkpoint.json.tmp.<pid>.<seq>.<nonce>` (réclamé exclusivement avec
   `open(..., "x")`, fsync) puis `os.replace` vers `checkpoint.json` **en
   dernier** ; fsync du répertoire de checkpoint après le replace.
   `checkpoint.json` est le seul point de commit.

Invariants de sûreté :

- Un fichier référencé par le manifest actuellement commité n'est **jamais**
  écrasé avant que le nouveau manifest ne soit commité (les générations sont
  uniques par nom et réclamées exclusivement ; `os.replace` sur un artefact de
  génération existant est interdit).
- Le manifest est sérialisé avec `json.dumps(..., allow_nan=False)` et
  pré-validé (preflight) avant la création de tout artefact : NaN/Inf ou valeur
  non-JSON sont refusés avant toute écriture.
- Le temp de manifest est **possédé explicitement** : chaque tentative écrit
  sous un nom unique (`checkpoint.json.tmp.<pid>.<seq>.<nonce>`) réclamé en
  exclusif, et le nettoyage ne supprime **que** le temp créé par la tentative
  courante — jamais le temp d'un autre writer/processus.  Cela élimine la
  course concurrente où le nettoyage d'une tentative perdante supprimait le
  temp de manifest du gagnant.
- En cas d'échec avant le `os.replace` du manifest : le manifest précédent et
  tous ses fichiers restent **byte-identiques et utilisables** ; les
  fichiers créés par la tentative courante sont nettoyés au mieux (jamais un
  chemin préexistant ni un artefact d'un autre writer).  Le nettoyage est
  structuré autour d'un drapeau explicite `manifest_committed` : un échec après
  le replace ne peut pas « rollback » les fichiers nouvellement référencés.
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
- **Pic mémoire réel** : le snapshot n'est pas limité à « six tableaux
  possédés ».  `_snapshot_channels` possède six copies float32
  (`6 × H × W × 4` octets) **en plus** des six buffers moteur vivants, et
  `_npy_bytes` sérialise chaque artefact dans un objet `bytes` de la taille
  d'un fichier `.npy` complet (`H × W × 4 + en-tête`).  Le pic est donc ≈
  `13 × H × W × 4` octets (6 buffers moteur + 6 snapshots + 1 `bytes`), hors
  marge interne de `np.save`.  Le flux mémoire n'est pas refondu en D1.
- La **reprise** (lecture/restauration/activation de Resume) n'est **pas**
  implémentée en D1 : le writer ne lit jamais, ne finalise jamais, et
  `_build_startup_refusal`/`_validate_resume_headless`/Qt refusent toujours le
  Resume Drizzle comme aujourd'hui.

### 5.4 Seam de continuation-writer (D2B1)

La factory `DrizzleCheckpointWriter.from_validated_result(result)` est le
**seul** point d'entrée en mode continuation.  Le `__init__` public reste
fresh-run-only (refus exact D1 de tout namespace non vide, sans bypass public
`allow_existing`).  Contrat :

- **Confiance minimale** : seuls deux champs du `result` fourni sont consommés —
  le `source_output_dir` (provenance immuable/normalisée) et la `generation`
  (jeton de fraîcheur/staleness).  Tous les autres payloads (manifest / session
  / counters / config / WCS / accumulateurs) sont **ignorés** : ils sont figés
  superficiellement (shallow-frozen) et donc mutables/tamperables.
- **Relecture fraîche** : la factory effectue un
  `read_drizzle_checkpoint(source_output_dir, require_exact_versions=True)`
  complet en interne, puis exige que la génération relue égale
  `result.generation` (sinon refus « stale », fermé : un autre writer a déjà
  continué).  Tout l'état de continuation (config / WCS / grille / session /
  ledger / compteurs / total d'exposition par canal / accumulateurs) est lié
  depuis **cette lecture fraîche**, jamais depuis le `result` fourni.
- **Objet de ré-arm dédié** : la factory retourne un `DrizzleContinuation`
  (figé) portant le `writer` frais **et** les `accumulators` reconstruits frais,
  plus `session` / `counters` / `completed_sources` / `generation` /
  `next_source_index` frais.  Le lifecycle doit continuer en mutant
  `continuation.accumulators` puis en appelant `continuation.writer.commit(...)`
  — il ne peut pas continuer par accident depuis les payloads stales/tamperés du
  `result` d'origine.
- **Aucune écriture / GC au ré-arm** : ré-armer (et tout refus) ne produit
  aucune écriture ni garbage-collection ; le dernier manifest commité reste
  l'autorité.

**Monotonie de continuation** (exécutée avant toute écriture, dans le try du
commit, donc refus = génération précédente byte-identique) :

- la liaison de session doit être identique à la génération chargée ;
- `frame_count` doit croître strictement (préfixe strictement plus long) ;
- `total_exposure_seconds` ne doit pas reculer ;
- **vérité cumulative, pas seulement `frame_count` / exposition totale** :
  - `exposure_unknown_count` ne doit pas décroître ;
  - **arithmétique connu/inconnu cumulative** : avec `delta_frame =
    new_frame - loaded_frame` et `delta_unknown = new_unknown -
    loaded_unknown`, on exige `0 <= delta_unknown <= delta_frame` et
    `known_added = delta_frame - delta_unknown` ; si `known_added == 0`
    (toutes les nouvelles poses sont inconnues), `total_exposure_seconds` /
    `exposure_min` / `exposure_max` doivent rester **exactement** inchangés
    (y compris `None`) — aucune fabrication/réécriture du résumé cumulatif ;
    si `known_added > 0`, `total_exposure_seconds` doit croître strictement et
    `exposure_min` / `exposure_max` doivent être connus après le commit ;
  - si `exposure_min` / `exposure_max` chargés sont connus, le nouveau min ne
    peut ni augmenter ni disparaître et le nouveau max ne peut ni diminuer ni
    disparaître ; une valeur chargée `None` **peut** devenir connue quand des
    poses connues arrivent ensuite (transition sémantiquement légale) ;
  - le `total_exptime` natif **de chaque canal** doit croître strictement quand
    `frame_count` croît et ne jamais reculer (les totaux par canal chargés sont
    capturés dans l'état de continuation et les snapshots sont validés avant
    écriture) ;
- le ledger complété doit conserver le ledger chargé comme préfixe exact.

**Avancée du jalon monotone** : après chaque commit de continuation réussi, le
writer met à jour son jalon interne (generation / session / counters / ledger /
totaux par canal) vers la génération qui vient d'être commitée.  Le même writer
ne peut donc jamais commiter ``N+2`` avec des compteurs / ledger reculés par
rapport à ``N+1``.

**Aucun travail faillible après le commit du manifest** : l'état de continuation
suivant (deep copies de session / counters / ledger + totaux par canal) est
construit **entièrement pendant le préflight**, avant toute écriture d'artefact
et avant le commit du manifest, puis stocké dans une locale.  Après un
`_write_manifest` réussi, seules des affectations scalaires / de référence non
faillibles sont autorisées ; le GC reste best-effort.  Un échec d'allocation
(ex. `MemoryError`) pendant la préparation de l'état suivant survient donc
**avant** toute écriture et laisse la génération N byte-identique.

**Provenance de répertoire exacte (`realpath`)** : `DrizzleCheckpointResult`
lie `source_output_dir` au **chemin réel canonique** (`os.path.realpath`), pas
seulement à un chemin absolu ; une racine symlink est résolue une fois à la
validation, de sorte qu'un re-pointage ultérieur du symlink ne peut pas
re-lier un résultat validé à un autre run.  La factory re-résout le chemin réel
et refuse si la provenance ne résout plus vers le répertoire validé (swap /
re-pointage de symlink) — elle ne lie jamais l'autre checkpoint.

**Limite connue (résolue en D2B2A, §5.5)** : la politique de *source move /
re-stat* du cycle de vie (quand les poses sources sont déplacées/archivées
après leur acceptation puis re-statées lors d'un `read_drizzle_checkpoint`
ultérieur) n'est **pas** modifiée par D2B1.  Le lecteur D2A re-state déjà
strictement (path / size / mtime_ns), et D2B1 ne touche ni au `queue_manager`,
ni à la GUI, ni au lifecycle, ni à la politique de déplacement des sources.
Cette coordination source-move ↔ re-stat est traitée en D2B2A (§5.5) par un
seam de résolution **opt-in** sans jamais affaiblir le lecteur strict par
défaut.

### 5.5 Seam de résolution de source (D2B2A)

Le lecteur D2A re-state strictement chaque identité à son chemin original
persisté (path / size / mtime_ns), donc il ne peut **pas** reprendre après un
`move_stacked=True` de production.  D2B2A ajoute un **seam de résolution
explicite et opt-in** sans changer le comportement par défaut :

- **Défaut inchangé** : `read_drizzle_checkpoint(output_dir)` (sans
  `resolver`) reste strict D2A — toute source déplacée / renommée / manquante /
  modifiée échoue fermé exactement comme avant.
- **`resolver` explicite** : `read_drizzle_checkpoint(output_dir,
  resolver=...)` offre chaque identité canonique plus un contexte
  (`role` / `index` / `is_completed` / `output_dir` / `input_roots`) au
  resolver, qui retourne une **liste ordonnée de chemins candidats**.  Le
  lecteur re-state lui-même **chaque** candidat et n'accepte qu'un fichier
  régulier non-symlink dont la taille + `mtime_ns` correspondent exactement —
  jamais une affirmation du callback.
- **Politique de production** : `SafeStackedSourceResolver(stacked_subdir_name)`
  (objet **immutable / gelé**) ne retourne que (a) le chemin original ou (b) le
  pendant déterministe `<original_dir>/<stacked_subdir_name>/<basename>` — la
  destination exacte de `tools.file_ops.move_to_stacked`.  Aucune recherche de
  répertoire, glob, fallback basename-seul, remap sans hash, ou fichier
  arbitrairement renommé ; un nom de collision `_dup_<timestamp>` n'est jamais
  deviné.
- **Injectivité / ordre** : la résolution préserve l'ordre du plan et refuse
  toute ambiguïté/duplication de **deux identités canoniques distinctes**
  résolvant vers un même chemin disque.  La **répétition légitime de la même
  identité canonique** — la référence d'alignement étant aussi l'une des
  observations du plan (même path + size + mtime_ns) — est acceptée et résout
  vers le même chemin sans ambiguïté.
- **Résultat exposé** : `DrizzleCheckpointResult` expose `resolved_reference`,
  `resolved_plan_paths`, `resolved_completed_paths`, `resolved_remaining_paths`
  et porte `resolution_policy` (la politique immuable, ou `None` en strict).
  Les identités persistées (manifest / session / ledger) restent **canoniques
  (chemins originaux)** — jamais de chemin `stacked/` réécrit.
- **Ré-arm sûr** : `DrizzleCheckpointWriter.from_validated_result` ré-applique
  `result.resolution_policy` sur sa relecture fraîche (jamais un callback
  mutable) et **ne lâche jamais** la validation des sources ; un callback
  arbitraire mutable est honoré pour la lecture immédiate mais **non porté**
  (le ré-arm retombe alors en strict, échec fermé).

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
  Inclut le refus de redémarrage (un second writer sur le même output refuse à
  la construction, avant toute écriture d'array, et le manifest + les six
  artefacts référencés restent byte-identiques) et le refus au premier commit
  si le namespace se matérialise après la construction.
- **C** — génération N→N+1 réussie : bascule atomique vers les seuls
  descripteurs N+1, génération courante jamais garbage-collectée.
- **D** — témoin d'ordre aux frontières sûres : pas d'écriture après 0 pose ou
  add échoué, pas de checkpoint entre canaux, cadence à `group_size`, force
  finale sur Stop/succès, idempotence, ledger/compteurs cohérents, échec = abort
  avant le déplacement de la source.
- **E** — échec fermé (mismatch config/empreinte, WCS/liaison de session absent,
  source dupliquée/non identifiable, buffers/compteurs non finis) sans nouveau
  manifest commité.  Inclut le **préflight de cohérence dépôt** : des
  accumulateurs dont `kernel` / `pixfrac` / `fillval` divergent des champs
  scientifiques canoniques sont refusés **avant** toute création du répertoire
  de checkpoint / de `run_config.cfg`.
- **F** — manifest auto-cohérent (ledger = préfixe ordonné du plan de longueur
  `frame_count`, `stacked_batches_count == frame_count`, `frame_count <=
  len(plan)`, identités strictes sans doublon), refus preflight NaN/non-JSON,
  et propriété de nettoyage : un artefact préexistant en collision n'est jamais
  supprimé par le nettoyage d'une tentative.
- **G** — propriété de **possession du temp de manifest** : une tentative ne
  supprime jamais un temp de manifest étranger (test direct), et un test de
  concurrence déterministe (barrières/événements) prouve que lorsque deux
  writers passent tous deux le refus initial du namespace vide, le gagnant
  réclame/possède son temp, le perdant échoue sur la collision d'artefact et
  nettoie, et le gagnant publie quand même — sans mutation d'octets étrangers.

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

### 7.3 Exécutés en D2A (lecteur read-only)

`tests/test_drizzle_checkpoint_reader.py` couvre :

- **A** — roundtrip d'inspection writer → reader (square et WHT Lanczos signé) :
  reconstruction bit-exacte des trois accumulateurs, validation
  config/digest/empreinte/WCS, `next_source_index` exact, et lecture **sans
  mutation** de l'arbre de checkpoint.  Inclut un test **non carré**
  writer → reader qui fixe la convention d'axes (`array_shape == (H,W)`,
  `pixel_shape == (W,H)`, artefacts `(H,W)`) avec contrôle de continuation
  bit-identique, et un test d'équivalence `fillval` numérique `0.0` vs chaîne
  canonique `"0.0"`.
- **B** — **continu == write/read/reconstruct/continue** bit-identique (SCI et
  WHT natifs) pour `square` / `gaussian` / `lanczos2` aux splits 2 (frontière
  de groupe), 3 (groupe partiel) et 5 (groupe partiel).
- **C** — ordre des sources / `next_source_index` exact.
- **D** — **matrice de corruption** (35 cas) : chaque classe échoue fermé avec
  `DrizzleCheckpointError` *avant* tout état restauré, et l'arbre de checkpoint
  reste **byte-identique** (snapshot avant/après) : troncature manifest, NaN,
  schéma inconnu, mauvais mode/état, génération invalide, mismatch
  config/empreinte/digest/version (drizzle et numpy), WCS vide/carte manquante,
  shape de sortie incohérent, traversée de chemin/symlink, artefact
  manquant/supplémentaire/génération mixte, taille/hash/dtype/shape falsifiés,
  tableau non fini, **divergence manifest-only des paramètres de dépôt de canal
  (`kernel` / `pixfrac` / `fillval`) vs config canonique**, source
  manquante/renommée/taille ou mtime modifiée, ledger dupliqué/désaligné,
  compteurs désalignés.

### 7.4 Contrat du lecteur D2A (`read_drizzle_checkpoint`)

- **Lecture seule, échec fermé** : ne mute jamais les octets du checkpoint, les
  sources, le répertoire de sortie ni l'état runtime vivant ; charge les
  tableaux avec `np.load(..., allow_pickle=False)` et retourne des copies
  float32 privées.
- **Politique de version documentée (continuation exacte)** : `schema_version`
  / `mode` / `state` doivent correspondre exactement ; `drizzle_lib_version`
  et `numpy_version` persistés doivent égaler les versions runtime, sinon
  refus (arrondi bit-identique).  Un relâchement éventuel vers un WARN est une
  décision D2B séparée.
- **Re-stat des sources** : la référence, chaque source du plan et chaque
  source du ledger complété sont re-statées (path/size/mtime_ns doivent
  correspondre) ; fichier manquant/renommé/modifié → refus.  Le ledger complété
  doit rester le préfixe ordonné exact du plan.
- **Reconstruction WCS/grid** : le WCS sérialisé est reconstruit, `array_shape`
  est rattaché, et le WCS doit **round-triper exactement** vers le dict de
  cartes persisté (contrat de grille de sortie exact).  Convention d'axes
  Astropy fixée : `array_shape == (H, W)` (ordre numpy) et
  `pixel_shape == (W, H)` (ordre FITS/NAXIS) ; `output_shape_hw` est `(H, W)`.
  Les artefacts natifs restent `(H, W)`.
- **Cross-check dépôt vs config canonique (échec fermé)** : après validation
  séparée de la config canonique / empreinte / digest et des entrées de canal,
  le lecteur exige que `kernel` / `pixfrac` / `fillval` de **chaque** canal
  égalent les champs scientifiques canoniques `drizzle_kernel_effective` /
  `drizzle_pixfrac_effective` / `drizzle_fillval`.  Une édition manifest-only
  des paramètres de dépôt (empreinte toujours valide) est donc refusée **avant**
  reconstruction, et non reconstruite avec de mauvais paramètres.
- **Règle d'équivalence `fillval` (documentée)** : `fillval` est comparé par
  équivalence scientifique/sérialisation — un `fillval` numérique et une chaîne
  qui parse vers le **même float fini** sont équivalents (ex. `0.0` == `"0.0"`
  == `"0.00"`) ; une chaîne non littéral-float (ex. `"INDEF"`) est comparée par
  identité de chaîne exacte et n'est jamais coercée en nombre.
- **Reconstruction des accumulateurs en dernier** : uniquement après validation
  complète, via `DrizzleAccumulator.from_native_state` et les paramètres
  runtime effectifs par canal (`kernel`/`pixfrac`/`fillval`/`total_exptime`) —
  aucun état restauré partiel visible.
- **Objet résultat explicite** : `DrizzleCheckpointResult` (manifest, session,
  compteurs, ledger complété, config, WCS, `output_shape_hw`, accumulateurs,
  `next_source_index`, génération) prêt pour le wiring lifecycle D2B.

### 7.5 Exécutés en D2B1 (seam de continuation-writer)

`tests/test_drizzle_checkpoint_continuation.py` couvre (39 cas) :

- **A** — régression du refus fresh-run D1 (un writer frais refuse toujours un
  namespace non vide, byte-identique) et absence de bypass public
  `allow_existing` ;
- **B** — le ré-arm est sans écriture / sans GC et retourne un
  `DrizzleContinuation` frais ; refus d'un dict arbitraire, d'un `result` stale
  (génération déplacée sur disque), d'un jeton `generation` falsifié ; et
  **ignorance des payloads mutés** du `result` fourni (manifest / session /
  counters / config / WCS / liste d'accumulateurs) — la factory relie l'état à
  la vérité disque non falsifiée ;
- **C** — cycles write → read → re-arm → continue → commit → read (``N+1`` et
  ``N+1 → N+2``) bit-exacts pour `square` et WHT Lanczos **signé** ;
- **D** — refus rollback / réordonnancement / préfixe de ledger divergent, plus
  la **vérité cumulative** : décroissance d'`exposure_unknown_count`, montée /
  disparition d'`exposure_min`, baisse / disparition d'`exposure_max`, rollback
  / divergence du `total_exptime` natif par canal, transition légale
  `None → connu` du min/max, et la **régression deux-commits sur le même
  writer** (``N+1`` commité puis rollback ``N+2`` refusé byte-identique ; une
  extension ``N+2`` valide réussit) ;
- **E** — matrice d'injection de panne (array / config / temp de manifest /
  replace) laissant la génération N byte-identique sans fichier de tentative
  résiduel ;
- **F** — deux writers de continuation en course depuis le même `result` :
  échec fermé, génération commitée cohérente ;
- **G** — **aucun travail faillible après le commit du manifest** : injection de
  panne dans la préparation de l'état de continuation suivant (et dans le
  `copy.deepcopy` sous-jacent) — l'échec survient **avant** toute écriture et
  laisse la génération N byte-identique sans artefact/temp de tentative ;
- **H** — **arithmétique connu/inconnu cumulative** : refus de l'inflation
  rétroactive d'`exposure_unknown_count` (`delta_unknown > delta_frame`), refus
  de la fabrication/modification de `total_exposure_seconds` /
  `exposure_min` / `exposure_max` quand toutes les nouvelles poses sont
  inconnues (`known_added == 0`), et transitions **valides** tout-inconnu et
  mixtes (connu + inconnu) ;
- **I** — **provenance de répertoire exacte** : un résultat validé via une
  racine symlink lie `source_output_dir` au chemin réel canonique ; un
  re-pointage ultérieur du symlink ne peut pas re-lier le ré-arm à l'autre
  checkpoint (le ré-arm reste sur le répertoire réel d'origine), et la factory
  refuse un swap de symlink de la provenance (skip propre si la plateforme ne
  supporte pas les symlinks).

### 7.6 Exécutés en D2B2A (seam de résolution de source)

`tests/test_drizzle_checkpoint_source_resolution.py` couvre (23 cas) :

- **défaut strict inchangé** : un lecteur sans `resolver` refuse une source
  complétée ou une référence déplacée vers `stacked/` ;
- **résolution opt-in** : `SafeStackedSourceResolver("stacked")` accepte un
  déplacement exact (taille + mtime préservés) de la référence et d'une source
  complétée, retourne `resolved_reference` / `resolved_plan_paths` /
  `resolved_completed_paths` / `resolved_remaining_paths` / `next_source_index`
  corrects et ordonnés, et laisse l'arbre byte-identique (lecture seule) ;
- **plan mixte** : sources complétées déplacées + sources en attente originales ;
- **matrice de refus byte-identique** : taille/mtime erronés, destination
  symlink, renommage arbitraire, mauvais `stacked_subdir`, collision
  `_dup_<timestamp>`, candidat dupliqué/ambigu (deux identités vers un même
  chemin), et retour de resolver invalide ;
- **référence = source du plan** : en défaut strict et avec
  `SafeStackedSourceResolver`, une référence qui est aussi la source 0 du plan
  (identité canonique identique, y compris déplacée vers `stacked/`) est
  acceptée sans erreur d'ambiguïté, avec `resolved_reference ==
  resolved_plan_paths[0]`, tranches ordonnées correctes et identités canoniques
  préservées ; le ré-arm de continuation reproduit la politique et commite
  encore les chemins originaux ;
- **ré-arm sans callback mutable** : la factory reproduit la politique
  immuable portée (`resolution_policy`) sur sa relecture fraîche, **ne lâche
  jamais** la validation des sources (une destination falsifiée après la
  lecture est refusée), et abandonne proprement un callback mutable (retombe en
  strict, échec fermé) ;
- **sous-classe mutable/écrasante non portée** : une sous-classe de
  `SafeStackedSourceResolver` qui ajoute un état mutable et surcharge la
  résolution est honorée pour la lecture immédiate mais **jamais portée**
  comme `resolution_policy` (le portage exige le **type exact** livré, pas une
  sous-classe), si bien que le ré-arm après une source déplacée retombe en
  strict et échoue fermé ;
- **identités canoniques préservées** : le writer de continuation commite
  toujours les chemins originaux du plan/ledger, jamais un chemin `stacked/`
  réécrit ;
- **continuation bit-exacte** : `square` et `lanczos2` (WHT signé) restent
  bit-identiques à travers un déplacement + ré-arm + deux commits de
  continuation ;
- **contrat de la politique** : `SafeStackedSourceResolver` est immuable,
  refuse un `stacked_subdir_name` invalide, et ne devine jamais un basename
  `_dup_`.

## 8. Inventaire de la mission de reprise (D2) — reader + continuation-writer + source-resolution seam faits, activation lifecycle NON implémentée

La fin de D1 ne complète pas RSM2 ; D2A livre le lecteur/validateur read-only
et D2B1 livre le seam de continuation-writer (ré-arm monotone depuis un résultat
validé), et D2B2A livre le seam de résolution de source opt-in (politique
original-ou-`stacked/` déterministe, immuable, portée comme provenance).  Il
reste à **activer la reprise** dans le lifecycle (D2B2,
**volontairement non touchée par D1/D2A/D2B1/D2B2A**) :

- ✅ lire/valider `checkpoint.json` et vérifier SHA-256/taille de chaque
  artefact avant réinjection (fait en D2A) ;
- ✅ reconstruire les trois accumulateurs via
  `DrizzleAccumulator.from_native_state` (fait en D2A) ;
- ✅ revalider la liaison de session et le ledger, refuser tout fichier
  modifié/renommé (fait en D2A) ;
- ✅ ré-armer le writer de continuation à ``N+1`` depuis un résultat validé,
  avec relecture fraîche, objet de ré-arm dédié et monotonie cumulative
  (fait en D2B1) ;
- ✅ résoudre opt-in les sources complétées / référence déplacées vers
  `stacked/` via la politique immuable `SafeStackedSourceResolver`, sans
  jamais affaiblir le lecteur strict par défaut ni réécrire les identités
  canoniques, en restant **injectif pour les identités distinctes** (deux
  identités canoniques distinctes résolvant vers un même chemin restent
  refusées) tout en acceptant la **répétition légitime de la même identité**
  (référence d'alignement également présente dans le plan) (fait en D2B2A) ;
- réaffirmer `FINALIZATION_MODE_DRIZZLE` et reprendre strictement après
  `last_processed_index` ;
- activer le chemin Resume Drizzle dans `_build_startup_refusal` /
  `_validate_resume_headless` / `_validate_and_open_resume` / Qt readiness —
  chemins **volontairement non touchés** par D1, D2A, D2B1 et D2B2A ;
- coordonner la politique *source move / re-stat* du lifecycle avec la reprise
  (le **seam** de résolution est livré en D2B2A ; le **wiring** lifecycle reste
  reporté à D2B2 et non modifié).
