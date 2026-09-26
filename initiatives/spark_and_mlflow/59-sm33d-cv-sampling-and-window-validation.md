# SM-33D: Core CV, training sampling and independent source windows

Date: 2026-09-25. Baseline: `fe3897c4`, branch `090`.
Status: implemented and verified locally; uncommitted. No cloud deployment.

## Delivered behavior

- Optional Basic-model CV uses `StatefulEstimator.cross_validate` and
  `FeatureEngineerFoldAdapter` on raw training rows. It supports pandas/Polars,
  K-fold, stratified, shuffle and time-series methods with 2–20 folds. Fixed
  model parameters survive unchanged; each fold relearns preprocessing.
- The final pipeline is fitted independently on all outer training rows. Final
  holdout rows never enter CV or preprocessing fit. Real MLflow tests prove
  CV on/off preserves final predictions and that temporal metadata is absent
  from the saved artifact's input columns.
- Workflow fields are `cv_enabled`, `cv_folds`, `cv_type`, `cv_shuffle` and
  `cv_random_state`. Defaults are disabled, 5, K-fold, true and 42. Time-series
  uses the normalized selected-window event column, requires no shuffle and
  rejects ties across a fold boundary. Unsupported methods/combinations fail;
  no nested-search claim is made.
- `cross_validation.json` records folds, aggregates, refit counts, dataset
  identity and engine in the candidate's MLflow run. `cv_*` aggregate metrics
  remain separate from the `heldout_*` promotion metrics.
- Optional `training_sample_rows`/`training_sample_seed` select up to N eligible
  source records by deterministic seeded SHA-256 ordering on Spark, before
  local transfer. Null leaves overflow-fail behavior intact. The sample includes
  the final holdout: 10,000 input rows at 20% test means 8,000 train / 2,000 test.
  This is not class-balanced sampling and never changes scoring completeness.
- Sampling validates source identities and eligible labels, then records
  source/eligible/selected counts and membership evidence. Sample and holdout
  identities are bound into saved training evidence and checked on approval.
- `training_window_mode` separates `full_snapshot`, `fixed_window` and
  `rolling_calendar` from random/temporal evaluation and cron. Random splitting
  can use an explicit observation window. Rolling selection requires an IANA
  `window_timezone`; temporal lookback includes the last-month holdout.
  `train_monthly` pins the latest version for every mode, but only rolling mode
  derives new observation boundaries. Availability cutoff remains independent.

## Source map

`local_cv.py` contains the focused CV contract and Core adapter. Existing
`local_retraining.py` owns source sampling, outer split and MLflow evidence;
`local_workflow.py` owns calendar derivation and dispatch. `workflow_config.py`
shares validation with these services. Generated workflow JSON exposes editable
defaults. Guided initialization sections remain SM-33E; no search/Optuna runtime
was added (SM-36).

## Verification

- 274 native integration tests passed: real local MLflow training/approval on
  both engines, fixed Random Forest parameters, CV fold isolation, temporal
  feature removal and identical final predictions with CV on/off; window/zone
  boundaries; sampling membership and existing lifecycle/operator regressions.
- 41 real Databricks CLI v1.17.0 generation tests passed using the existing
  `skyulf` profile. Both engines/tasks and published examples validate locally.
- 14 real local WSL Spark/Delta tests passed. A 100,000-row source yields exactly
  10,000 selected rows; layout/partition changes preserve keys; changing seed
  changes keys; replay uses the pinned version. Late outcomes are excluded
  before sampling, duplicate keys and NaN targets fail. Existing source date,
  timezone, DST and transfer-bound checks also pass.
- Independent read-only review found no actionable defects in the final
  CV/isolation, sampling/replay and calendar contracts.
- Full repository ty, scoped Ruff/format, strict MkDocs and current wheel build
  passed. The generated serverless dev project passed `bundle validate --strict`
  with the newly built wheel; no deployment or remote job run followed.

The first tests failed for missing CV support/MLflow reports and missing source
selection fields as intended. New Delta coverage exposed a business key named
`count` colliding with Spark's default aggregation name; a collision-safe helper
name and explicit numeric-NaN validation fixed it. Go-template unit fixtures were
updated for the new conditional defaults; real CLI rendering also passed.

Windows/WSL notes: native tests use a workspace `.cache/` basetemp. WSL uses
`.cache/sm15-linux-run.sh` and `-o addopts=` because that environment does not
install the root benchmark plugin. Wheel build uses a workspace uv cache. The
built wheel needed its parent-folder ACL inheritance restored for CLI validation
under the normal user; no source permissions were broadened.

## Boundaries and next step

Local checks and Bundle validation are not cloud execution evidence. SM-33E is
next: guided basics/preprocessing/model/CV setup and combined personal-serverless
train/MLflow/approval/score acceptance. SM-34 stays WAIT until then.

Earlier experimental candidate evidence must be recreated. New sampling fields
participate in the dataset identity even when sampling is disabled, so old
evidence is not silently upgraded. This follows the user's explicit
pre-production decision to avoid compatibility readers and aliases.

Sampling limits input transfer, not distributed scans or total process RAM.
Validation/count/hash ordering can scan the selected source multiple times.
Temporal CV operates on row folds; shared timestamps across boundaries fail
instead of being split into both train and validation. Custom/group-aware CV,
advanced tuning and custom FE scenarios remain their separately queued work.
