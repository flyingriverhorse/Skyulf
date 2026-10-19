# Configure this project's preprocessing

YAML selects ordered steps; Python supplies custom factories. Fitted state and
selected source are saved with the model. Prediction does not reread edited files.

## Files to edit

| File | Purpose |
| --- | --- |
| `config/pre_split.yml` | Fixed normalization and eligibility before splitting; no learned statistics. |
| `config/preprocessing.yml` | Transformations fitted on training rows inside each fold. |
| `config/training.yml` | Select `pre_split_recipe`, `preprocessing_recipe`, inputs and target. |
| `src/features/pre_split.py`, `preprocessing.py`, your own modules | Top-level factories selected by a recipe's `custom` path. |
| `config/scoring.yml` | Saved scoring eligibility and business outputs; custom callbacks stay in `src/features/scoring.py`. |
| `config/features.yml`, `src/features/groups/` | Separate optional upstream Spark feature production. |

Both phase YAML files require `version: 1`, `recipes`, and a `default` recipe.
Generated `default`/`none` recipes are empty; `example_` recipes run only when
selected. Do not add Python builders for a YAML-owned phase or inline steps in
`training.yml` > `pipeline.preprocessing`/`pipeline.modeling`.

For example, add this step to a selected preprocessing recipe:

```yaml
- custom: preprocessing.log_feature
  params: {column: income}
```

This calls `src/features/preprocessing.py`'s factory `log_feature(column="income")`.
The factory returns one Core step; its callback later receives the rows.
Use `column_step` for fixed columns, `fitted_step` for learned state and
`filter_step` for fixed row eligibility. A bare DataFrame-returning function is
not a valid factory. Built-in steps use `name`, `transformer` and `params`.

## Commands and critical rules

```powershell
python src/tools/smoke.py
python src/tools/preview.py --action train
python src/tools/preview.py --list-preprocessors
python src/tools/preview.py --action score
```

Smoke checks declarations/syntax without running factories. Preview resolves
trusted factories but does not fit models or read cloud data. Keep I/O out of
imports/factories. Preserve record keys and protected time/group/weight metadata.
Declare package assets and exact dependency pins; changing them needs a new fit.

Scoring defaults to saved pre-split reuse. A target-reading filter requires an
explicit `skip_target_pre_split_steps: true` in `config/scoring.yml` or deliberate custom scoring mode
before training. That skip removes the whole target-reading step; keep feature
and target filters separate when feature checks must remain.

Upstream `groups/` code is not captured with model recipes. Keep model functions
in phase/custom modules and do not import feature-group producers from them.
Custom local examples do not establish Spark-worker or endpoint admission.
Spark and online-feature projects lint and probe `column_step`/`fitted_step` code
in training; check it first with `python src/tools/preview.py --action train` (see
https://www.skyulf.com/manual/user_guide/databricks_spark_scoring.html#when-the-probe-fails).

## Detailed instructions

- [All recipes, step cards and six-step customer example](https://www.skyulf.com/manual/user_guide/preprocessing_recipes.html)
- [Write custom transformations](https://www.skyulf.com/manual/user_guide/databricks_custom_steps.html)
  and [fixed pre-split filters](https://www.skyulf.com/manual/user_guide/databricks_custom_pre_split.html)
- [Saved-state diagnostics and continued history](https://www.skyulf.com/manual/user_guide/preprocessing_context.html)
- [Feature groups and upstream joins](https://www.skyulf.com/manual/user_guide/databricks_features.html)
- [Scoring eligibility and optional prediction publication](https://www.skyulf.com/manual/user_guide/databricks_scoring.html)

## Version note

Public links show the latest manual; [README.md](README.md) explains how to match
them to this project's pinned Core release.
