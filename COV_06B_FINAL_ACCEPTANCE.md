# COV-06B Final Acceptance

## Verdict

**ACCEPTED**

## Reviewed state

- Branch: `feature/coverage-aware-peripheral-reconstruction`
- Starting SHA: `7d9306589998fa58e550f0167f4cb4b89ba95eb0`
- Implementation SHA: `adb98eeffb25291a6c7bdc2f60f7e148144faca8`
- Nono-reviewed HEAD: `b0134f37c287125ee8a71aeba636be9381ffff1e`
- Nono-review report commit: `debc5b9a45170d7c2cb40ca0867a1ad47033ea08`
- Original baseline / `origin/main` / `origin/beta`:
  `501eb9b5031a1ea81f09e6f01687a4cc349de879`

The final acceptance document is committed after the review-report commit.
There were no production-code or test changes after Nono reviewed
`b0134f37c287125ee8a71aeba636be9381ffff1e`; the only later repository changes
are the independent review and this acceptance evidence.  A second Nono review
is therefore not required by the mission rule.

## Independent review

Nono verdict: **ACCEPT**.

Nono answered N1--N18 with repository and test evidence:

- old persisted `apply_feathering=true` migrates once to OFF;
- no old unversioned setting can silently reactivate the legacy path;
- deprecated controls are absent from normal Qt and explicitly isolated in Tk;
- **Coverage support taper** reflects the current mixed reducer semantics;
- the validated taper remains default ON with unchanged mathematics;
- coverage-aware final reconstruction is visible, persistent, propagated and
  default OFF;
- render OFF/ON leaves scientific SCI/WHT/support/WCS unchanged;
- positive Drizzle support remains separate from signed native WHT;
- no reducer, HSI, resume, registration or scientific weighting contract
  changed;
- benchmark arithmetic, baseline-failure classification, scope and repository
  hygiene are correct.

Nono reported no critical, high or medium findings.  Its three informational
notes (an internal radial-feathering docstring, file-layer legacy-copy behavior,
and deliberately retained/deprecated Tk control) require no COV-06B change.

## Junior verification and acceptance rationale

Junior independently inspected:

- the actual branch, commits, status and full COV-06B diff;
- the migration implementation and both Qt/Tk loading paths;
- the Qt state/serialization/backend chain;
- the final-render placement and A/B scientific-equivalence test;
- the Nono report and its claimed reviewed HEAD;
- the final tracked worktree and `git diff --check`.

The closure target is satisfied:

```text
old persisted Feathering=true
 -> versioned one-time migration
 -> legacy inverse-WHT Feathering OFF

fresh settings
 -> coverage support taper ON
 -> coverage-aware final reconstruction OFF

Qt render checkbox
 -> persisted state
 -> backend
 -> final cosmetic render only

render OFF / ON
 -> scientific FITS invariant
```

## Validation summary

Junior aggregate, excluding repeated focused reruns:

```text
739 passed
1 skipped
6 known starting-HEAD failures
0 COV-06B regressions
```

The six known failures (four resume tests, one save-final-stack dummy test and
one obsolete Boring source-text assertion) were reproduced at starting SHA
`7d9306589998fa58e550f0167f4cb4b89ba95eb0`.  Nono independently reproduced
the same failure set at that starting SHA.

Final COV/Qt/settings/backend focused group:

```text
290 passed, 1 skipped, 0 failed
```

Established HSI group:

```text
150 passed, 0 failed
```

Compilation and whitespace checks are clean.

## Repository policy

- No merge to main or beta.
- No push or remote branch update.
- No tag, release, deployment or version bump.
- Coco was never reset, contacted or used.
- Nono was the sole delegated reviewer.
- Pre-existing untracked files `COV_NONO_REVIEW.md`, `FETCH_HEAD` and
  `main` remain preserved and explained.
- The real-data 1602-image OFF/ON witness is explicitly deferred to Tristan.
