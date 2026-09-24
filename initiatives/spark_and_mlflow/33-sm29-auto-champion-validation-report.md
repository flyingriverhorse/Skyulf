# SM-29: Automatic champion validation

Date: 2026-09-24. Branch: `090`.

Live closure: **PASS** for the selected Polars/serverless workflow. See
[SM-27/SM-29 live evidence](35-sm27-sm29-live-validation-report.md).

## Local result

The Bundle offers `pinned_version` and `auto_champion`. Automatic mode
compares a registered candidate with champion on a pinned temporal holdout,
checks the chosen metric and thresholds, stages and promotes an eligible
candidate, and calls the existing score job. First champion initialization
requires an absolute threshold and a fresh heldout check. Score resolves
champion once to a concrete version per run. The train and score jobs remain
serial and queue concurrent requests. Manual mode retains its version pin.
Alias mutation first records a pending model tag. Automatic training and
scoring reject pending tags and existing aliases without a controlled
committed receipt, so an uncertain registry write cannot become a silently
accepted champion on retry.

No alias control table was added. Automatic alias writes rely on exclusive
model-alias write privileges for the train job identity. The caller must
enforce that boundary before production use. Registry alias promotion and
Delta publication are separate transactions; a failed score needs retry,
not a silent alias rollback.

## Verification

- The focused Bundle, retraining, MLflow registry, comparison and real local
  promotion suite passed **94 tests**. First-champion tests exercise a
  verified receipt, an existing alias without a receipt, an unknown alias
  outcome, and rejected missing/failed quality thresholds.
- Databricks CLI 1.17.0 generated a manual serverless Bundle and a Polars
  `auto_champion` full-rebuild Bundle with `heldout_rmse`, minimum improvement
  `0.05`, quality threshold `10.0`, and a custom paused monthly cron. Both
  passed strict read-only `dev` validation with the `skyulf` profile.
- A fresh `uv build --wheel --no-build-isolation` succeeded. Focused Ruff
  check/format, ty, `git diff --check` and `mkdocs build --strict` passed.
  The CLI projects were generated from the final template and validated
  read-only. The separate approved rehearsal subsequently deployed and ran
  the generated project as recorded in the live report.

## Live validation completed

The isolated personal-workspace rehearsal passed first champion, improved
v2, ineligible v3, automatic train-to-score handoff and both scoring modes.
It verified model-version metadata, retained v1 output, v2 view activation,
queued no-op scores, expected score failure/retry and a restricted principal's
real alias-write denial. Production still requires an exclusive serialized
alias-writer identity; the personal test does not prove every administrative
principal is excluded. Company targets remain unconfigured and untested.
