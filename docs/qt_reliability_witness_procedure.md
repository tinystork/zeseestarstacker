# ZSSS Qt Reliability — Procédure witness courte

État: **QT_RELIABILITY_WITNESS_READY**
Repo: `~/projects/zeseestarstacker`, branche `feature/pyside6-migration`
Baseline testée: HEAD `b6ac30d` + corrections R1→R3b + D3 + lifecycle (non commitées, working tree)

## 1. Lancer l'application Qt

```bash
cd ~/projects/zeseestarstacker
.venv/bin/python -m seestar.gui_qt.app        # (ou la commande GUI habituelle)
```

## 2. Dataset

Le witness humain peut réutiliser son dataset M16 habituel, ou les copies fraîches
du master de test:

```bash
cd ~/projects/zeseestarstacker
.venv/bin/python fixture/copy_fresh.py /tmp/witness_in 6   # copie 6 images fraîches
```

**Règle:** toujours une copie fraîche par run — les runs ne doivent plus déplacer
les sources (invariant `move_stacked=False` → zéro mutation).

## 3. Matrice minimale (onglet Empilement)

| # | Workflow | Réglages |
|---|----------|----------|
| 1 | Classic | Enable drizzle OFF, Combinaison finale = Mean |
| 2 | Reproject | Combinaison finale = Reproject |
| 3 | Reproject & Coadd | Combinaison finale = Reproject & Coadd |
| 4 | Drizzle Standard ×2 | Enable drizzle ON, Mode = Standard, Scale = ×2 |
| 5 | Drizzle Large dataset ×2 | Enable drizzle ON, Mode = Large dataset, Scale = ×2 |
| 6 | Drizzle Standard ×3 | Enable drizzle ON, Mode = Standard, Scale = ×3 |

Le bloc Drizzle de l'onglet Empilement doit montrer, dans l'ordre Tk:
`Enable drizzle → Mode (Standard/Large dataset) → Preview group size → hint →
Scale (×2/×3/×4) → WHT Threshold % → Kernel → Pixfrac`. Rien dans Expert.

## 4. Ce qu'il faut constater

- **Résultat:** chaque run se termine proprement (SUCCESS, ou EMPTY/NO OUTPUT si
  aucun output — jamais un simple "Finished." vide).
- **Shape finale** (dans le dossier de sortie, `stack_final.fit`):
  - Classic/Reproject: ~1920×1080 (taille native)
  - Drizzle ×2: ~3840×2160 (2× par axe)
  - Drizzle ×3: ~5760×3240 (3× par axe)
- **Pixel scale WCS** (si solveur configuré):
  - ×2 → ~1.19"/px (≈ natif/2), ×3 → ~0.79"/px (≈ natif/3)
- **Sources:** aucun `.fit` déplacé dans `stacked/` ni `unaligned_by_stacker/`
  (sauf vrai échec d'alignement).
- **Second run:** la barre de progression repart de 0, pas de résidu du run précédent.

## 5. Vérification automatisée (optionnelle)

```bash
cd ~/projects/zeseestarstacker
TMPDIR=/tmp/witness_tmp QT_QPA_PLATFORM=offscreen .venv/bin/python \
  /home/tristan/.openclaw/workspace/review/zsss-qt-reliability-matrix/matrix.py
```

produit `matrix_results.json` avec les 6 workflows, shapes, pixel scales,
mutation et erreurs. (Attention: le moteur écrit ses memmaps dans `$TMPDIR`;
utiliser un dossier avec assez d'espace, pas le tmpfs système plein.)

## 6. Règles

- NO PUSH / NO MERGE / NO TAG / pas de travail sur l'entry point M26 tant que
  le witness humain n'a pas validé.
- Les corrections en cours sont dans le working tree (non commitées) : D3 UI
  drizzle + lifecycle SUCCESS/EMPTY.
