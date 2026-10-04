# SM-36b follow-up: independent feature recipes and asset help

Date: 2026-09-29. Status: DONE; local checks and cloud acceptance passed.
Delivered with SM-36b. This extends [report107](107-sm36b-training-branches.md).

## User contract

Keep custom implementations in one shared `src/features/custom/` package. Define
named step lists in `features/preprocessing.py` and `features/pre_split.py`.
Each entry in `modeling/branches.py` selects `preprocessing_recipe` and
`pre_split_recipe` independently. Omitted selectors preserve existing default
and zero-argument builders. Unknown names fail before source reads.

The starter provides `default`, `none`, `frequency_only`, `imputer_only` and
`combined` preprocessing recipes, plus `default`, `none` and `complete_inputs`
pre-split recipes. Default recipes remain inactive. Adapt recipe columns to the
actual branch inputs; custom implementation and step selection stay separate.
The commented branch starter includes `demand_ensemble`: voting regression with
Linear Regression and Ridge, explicit weights and its own recipe selectors.
Actual CLI-generated examples validate under both base task choices.

Selected factory arguments are bound into saved project source. Training,
replayed branch plans, saved-model inference and pre-split scoring reuse restore
the selected recipe, including its custom registrations. Learned state remains
specific to each fitted branch and training fold.

`assets.json` now supports an object containing `_help` and `files`; legacy lists
still load. Inline help explains declaring project-owned files, reading them via
`read_project_asset`, immutable delivery, dependency pins and size limits.
Help text does not change the asset snapshot. Assets are not automatically
executed steps or a substitute for learned Calculator state.

Multi-target initialization asks for shared source, keys, budgets, compute and
training schedule. Branch-owned model, target, search, CV, split, quality and
scoring configuration is edited in `branches.py`, avoiding redundant questions.

## Implementation and verification

- `project.py`, `_project_recipes.py`, `branch_notebook.py`: independent recipe
  selection and persisted factory bindings, with legacy compatibility.
- `_project_files.py`: documented asset manifests with unchanged path, size and
  dependency validation.
- Template feature/modeling files, schema, README and SDK guide: configuration,
  defaults and user instructions.
- Combined affected runtime suite: **188 passed, 1 skipped**. The skip requires
  Windows symlink privilege. Includes actual pandas/Polars fitting, saved-model
  and plan replay in fresh processes, source deletion and scoring reuse.
- Template/layout suite: **105 passed**, including actual CLI generation and
  generated project preflight. Conditional visibility is checked against the
  schema; an interactive terminal prompt session is not claimed.
- Full Ruff, format check (1,093 files), full CI Ty scope, backend/Core CCN <= 10,
  strict documentation build and `git diff --check` passed.
- Independent read-only review found no remaining concrete correctness issue.

### Final commit verification

The expanded affected selection passed 302 tests and skipped three (one Windows
symlink privilege case, two optional local Delta cases). Two new ensemble sample
checks initially failed because the test's uncommenting helper also stripped an
embedded explanatory comment. Moving that explanation outside the code sample
resolved both failures; the entire branch template module then passed **16 tests**,
including both real CLI-generated base task variants. No runtime change was needed.
All current Core Python files match the successful cloud wheel after line-ending
normalization. Pre-commit hooks passed, including Ruff, format, Ty and CCN 10;
frontend hooks correctly had no affected files. Full format scope: 1,093 files.

## Databricks acceptance

- Workspace: `dbc-45604623-c18b.cloud.databricks.com`, profile `skyulf`.
- Isolated path: `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_lifecycle_test/recipes_20260929_r1`.
- Wheel SHA256: `a168faa49ffbee80dfb596123860936fc1839a97eb9e3808597552b8fbe865fd`.
- Run `535676495601679`, task `43689010285191`: TERMINATED/SUCCESS,
  notebook `status=passed`, result not truncated; total duration 651,932 ms.
- [Run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/812050928200690/run/535676495601679).
- Uses the actual branch notebook adapter with generated configuration and real
  Databricks widgets, then registered-model scoring into real Delta tables.
  Four branches are checked on both pandas and Polars: frequency-only regression,
  imputation-only regression, grid-tuned classification with stratified CV, and
  a voting ensemble with a custom asset-backed preprocessing step.
- Scoring checks missing values, unseen categories, selected pre-split reuse,
  keyed excluded outcomes, idempotent second calls and fresh-process MLflow
  prediction parity after deleting editable feature sources/assets.

### Verified results

| Engine | Branch | Preprocessing | Pre-split | Predicted / excluded |
|---|---|---|---|---|
| pandas | a_frequency | frequency_only | none | 6 / 0 |
| pandas | b_imputer | imputer_only | none | 6 / 0 |
| pandas | c_combined | combined | complete_inputs | 5 / 1 |
| pandas | d_asset_ensemble | asset_combined | complete_inputs | 5 / 1 |
| polars | a_frequency | frequency_only | none | 6 / 0 |
| polars | b_imputer | imputer_only | none | 6 / 0 |
| polars | c_combined | combined | complete_inputs | 5 / 1 |
| polars | d_asset_ensemble | asset_combined | complete_inputs | 5 / 1 |

All eight models registered as version 1 under
`workspace.skyulf_lifecycle_test.recipes_20260929_ee161bcc_<engine>_<branch>`.
All eight second scoring calls were no-ops, and all eight isolated-process
reloads matched published predictions. Total output: 48 keyed rows, 44 predictions
and 4 explicit exclusions. Each branch excluded its own five missing labels
during training. Registered alias maps stayed empty.

MLflow parents: pandas `548a4aa16c42461ba7de339a044b0ddf`, Polars
`a7c8aaef75ba42c18ee36378d2a8b182`. Owned source, score-input and prediction
tables were dropped after assertions; model versions and MLflow evidence remain.
The first and only submitted run for this follow-up passed.

## Remaining boundaries

Distinct target models can each have a champion alias. This training flow leaves
all aliases unchanged and creates candidates only. Coherent model-set activation,
rollback and composed scoring remain **SM-36c**, the next READY item.
This acceptance exercises independent scoring outputs, not composed publication.
No persistent Bundle jobs are deployed or changed by the one-time test submission.
