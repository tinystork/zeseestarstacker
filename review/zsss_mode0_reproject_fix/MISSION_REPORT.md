# Mission ZSSS — Mode 0 Reproject Correctness + Field Rotation (2026-08-20)

Statut : TERMINÉ — root cause démontrée, correctif implémenté, tests OK, dataset réel validé, commit local `eefccc6` (aucun push).

## Contexte
- Restauration du mode `batch_size=0 / Classic Stacking / Reproject & Coadd` de ZeSeestarStacker
  (branche `beta`, HEAD b5d6363). ASTAP comme solveur. Pas de push. Pas de Drizzle. Pas de refactor.
- Symptôme : batches corrects (`stack_batch027.fit`, médianes RGB ≈ 0.0124, max 0.45/0.36/0.24)
  mais final `stack_final_classic_reproject_zm.fit` quasi uniforme (R≈65054, G≈65261, B≈65534 uint16).

## Root cause (démontrée numériquement)
1. **ASTAP v2026.03.20 écrit l'échelle 3× dans le `.wcs`** : `PC` (qui doit être sans dimension),
   `CDELT` ET `CD` portent tous ~2.37″/pix (vérifié sur `/tmp/astap_test/frame.wcs`, solve réel M16).
2. `WCS(hdr, relax=True, fix=True)` garde les trois → wcslib applique `PC × CDELT` → échelle effective
   ≈ `scale²` ≈ 0.00156″/pix : **tous les transforms pixel→monde sont écrasés** (le champ entier se
   replie sur ~1 pixel) et `proj_plane_pixel_scales` rapporte 0.0016″ → warning
   « Reference WCS pixel scale 0.002 arcsec/pix outside [0.1, 30.0]; clipping » (lignes 4721/4863/15273).
3. `_update_fits_header_with_wcs` → `to_header(relax=True)` écrit PC=scale + CDELT=scale (double encodage,
   sans CD) → le header FITS final (et le `reference_header_for_wcs`) est double-encodé.
4. **`_reproject_classic_batches_zm` hérite CDELT mais PAS PC** (liste de clés incomplète) :
   les batches reçoivent un WCS CDELT=1.0 (forme des frames M16 : PC=scale + CDELT=1.0) sans rotation
   → échelle 1°/pix. La reprojection de chaque batch (1°/pix) sur la grille de sortie (2.37″/pix)
   compresse l'image en <1 pixel → sortie ≈ constante ≈ fond (0.0124).
5. `_save_final_stack` (uint16, max→65535) mappe ce fond constant ≈ 0.0124 → ≈ 65 000 → « blanc quasi uniforme ».
   La moyenne est bonne ; la structure est détruite par le WCS de batch.

## Correctif (2 fichiers, minimal)
- `seestar/alignment/astrometry_solver.py` : `_canonicalize_wcs_scale()` appelé dans
  `_parse_wcs_file_content` — reconstruit PC (rotation sans dimension) + CDELT (échelle) depuis la
  matrice CD. Invariant de transform vérifié (≤1e-9° vs vérité CD pure). `to_header` redevient
  mono-encodé ; `proj_plane_pixel_scales` = 2.3727″ ; warning 0.002″ éliminé à la source.
- `seestar/queuep/queue_manager.py` : `_REFERENCE_WCS_KEYS` (PC1_1..PC2_2 + CDELT + CD + CTYPE/CRVAL/
  CRPIX/CUNIT/RADESYS/EQUINOX/LONPOLE/LATPOLE) utilisé dans les 2 boucles d'héritage WCS ;
  `_header_has_wcs_keywords()` accepte les formes CD **et** PC+CDELT (remplace les 2 checks inline).

## Audit rotation de champ (M16, 80×10s)
- Solutions ASTAP frames 1/20/40/60/80 : PA 178.231→178.244° → **rotation de champ totale ≈ 0.013°**
  sur la séquence (négligeable ici).
- Alignement : chaque frame est aligné sur la référence fixe (similarity scale=1.0, rotation+translation)
  → tous les batches partagent l'orientation pixel de la référence.
- WCS de référence : rotation 178.23°, échelle 2.3727″/pix. Après correctif, chaque batch hérite du
  WCS complet → même grille céleste → le Reproject final applique UNE SEULE fois la projection céleste
  (rotation + échelle). Classification : **Cas B/C hybride** — la derotation est préservée ; le
  Reproject final reste indispensable (il porte la rotation/échelle céleste, pas un second rééchantillonnage).
- Reproject final = transform identité grille→grille après correctif (les batches sont déjà sur la grille
  référence) : 1 seul resampling par batch, pas de perte additionnelle.

## Preuve numérique (harness fidèle, code réel `_reproject_classic_batches_zm`)
- AVANT : final uint16 R med=64910, G=65078, B=65317 (≈constant), WCS CDELT=1.0+PC=scale.
- APRÈS : final uint16 R med≈1850 (fond 0.0127×65535/0.45 ✓), max R=65535 (étoile 0.447 ✓),
  WCS CDELT=6.5886e-4 (scale) + PC sans dimension → échelle effective 2.3727″/pix.
- cov_sum [B1-COADD-FIX] ≈ 165 888 003 ≈ 27 batches × ~3 img × 2073600 px × radial(≈0.988) ✓
  (pas 80 inputs : la « somme × 80 » était une interprétation du fond constant).

## Tests
- Nouveau : `tests/test_reproject_zm_wcs_fix.py` (7 tests) — canonicalisation (échelle 2.3727″,
  mono-encodage, invariance 1e-9°), parse réel, `_header_has_wcs_keywords`, e2e mode-0 64×64
  (structure préservée, WCS final correct) — **7 passed**.
- Suite : 183 passed (49 fichiers, isolés). Échecs restants = PRÉ-EXISTANTS vérifiés sur HEAD propre :
  test_astap_wcs_padding, test_load_wcs_ignore_missing_simple, test_preserve_linear_output,
  test_quality_parallel, test_reproject_utils ×7).

## Historique Git pertinent
- c8029b9 (2025-09-18) : mode 0 → héritage WCS référence (liste CDELT-only) — origine du bug 4.
- 2a778d2 (2025-10-08) : garde `if bs_mode == 0: return False` + force_local + message fallback
  « Astropy a renvoye une image vide ».
- 06fad91 (2025-10-15) : `_ensure_reference_wcs_for_mode0` + white-fix + flux mode 0 (mort jusqu'à M2c).
- b5d6363 (2026-08-20, M2c) : suppression de la garde → chemin 06fad91 réactivé → bug visible.
- 82f2c19 / 367976f : origine détection « image vide » (blank detection) — fallback local OK,
  pas en cause (le chemin astropy produisait une image quasi constante, pas vide, dans le run réel).

## Validation dataset réel (27 batches, chemin final complet, ASTAP réel)
- Final fixé : uint16 (3,1920,1080), fond R≈1850 (≈0.0126 float), étoiles max R=65535 (0.447),
  286 détections étoiles, pic à (639,424) (région M16) — image exploitable.
- WCS final : CDELT=6.5886e-4 (scale), PC sans dimension (|PC1|=1.000006), échelle effective
  2.3727″/pix, rotation 178.231° (== solution ASTAP), CRVAL ASTAP. Warning 0.002″ éliminé.
- Timings final pass : avant (run GUI réel) ≈ 1241 s (415+403+423) ; après (harness 27 batches,
  même machine, même chemin) ≈ 807 s → -35 % (caveat : contexte harness vs GUI).
- Rotation de champ : séquence M16 PA 178.231→178.244 (0.013° total) ; alignement per-frame sur
  référence fixe ; batches tous sur la grille référence ; Reproject final = UNE projection céleste
  (rotation+échelle) — derotation préservée (Cas B/C hybride).

## Résultats tests
- Nouveau `tests/test_reproject_zm_wcs_fix.py` : 7 passed.
- 183 passed au total (49 fichiers, isolés). Échecs restants = pré-existants sur HEAD propre
  (test_astap_wcs_padding, test_load_wcs_ignore_missing_simple, test_preserve_linear_output,
  test_quality_parallel, test_reproject_utils ×7, + pollution d'ordre sys.modules par
  test_incremental_reprojection — vérifié identique sur HEAD propre).

## Commit local
- `eefccc6` "Fix mode-0 Reproject&Coadd: canonical WCS scale + full reference WCS inheritance"
  (2 fichiers + 1 test + review/zsss_mode0_reproject_fix/zsss_mode0_fix_evidence.zip).
- Aucun push (HEAD local = eefccc6, branche beta, 10 commits devant origin/beta).
