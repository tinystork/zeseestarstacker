# M3-D — Design de checkpoint / reprise (étude, sans implémentation)

Statut : **DESIGN ONLY**. Aucun code de checkpoint n'est implémenté dans cette
mission M3-D-3. Ce document fige l'état mathématique à préserver et les
contraintes de reprise, afin qu'une éventuelle mission ultérieure puisse
implémenter un checkpoint **sans changer la science** du drizzle M3.

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

- **Conteneur :** `.npz` (tableaux `_out_img`/`_out_wht` par canal) **ou** FITS
  multi-extension (2 HDU image par canal : OUTIMG/OUTWHT). `.npz` est plus simple
  et suffisant (pas de WCS à stocker dans le tableau lui-même, le WCS est dans le
  JSON).
- **Métadonnées :** `JSON` (manifest, géométrie, paramètres, compteurs,
  `schema_version`, `drizzle_lib_version`, checksums).
- **Écriture atomique :** écrire `checkpoint.tmp` puis `os.replace()` vers le
  nom final ; jamais d'écriture partielle visible.
- **Checksum :** sha256 sur le payload binaire (et éventuellement sur le JSON),
  vérifié à la reprise avant toute réinjection.
- **Compat versionnée :** refuser `schema_version` inconnue avec un message
  actionnable ; conserver `drizzle_lib_version` pour diagnostiquer une dérive de
  bibliothèque.

Exemple de layout :

```
{output_folder}/.m3d_checkpoint/
    checkpoint.json          # métadonnées + manifest + compteurs + checksums
    accum_ch0_outimg.npy     # ou une seule archive accumulators.npz
    accum_ch0_outwht.npy
    accum_ch1_*.npy
    accum_ch2_*.npy
```

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

## 7. Tests futurs recommandés (non exécutés en M3-D-3)

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
