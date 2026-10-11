# Feature folder layout and preprocessing boundary

2026-10-07. User requested that upstream feature-group code live inside the
existing generated `src/features/` directory, and asked which preprocessing
belongs upstream. This continuation preserves the earlier staged delivery182
changes; it does not stage, commit, push or deploy anything.

## Layout and behavior

- Spark producers now use `src/features/groups/module.py:function` in
  `config/features.yml`. The template emits a `groups/README.md` explaining
  how to add these optional modules. Empty groups still add no job.
- The existing `pre_split.py`, `preprocessing.py`, `scoring.py` and `custom/`
  entry points stay in place. Their comments now refer to training YAML.
- The base input defines observation rows/keys/time and optional labels. Each
  Spark group can create several columns and writes its own Delta output;
  merging attaches these values to the base observations.
- Fixed shared formulas and aggregations belong upstream. Learned means,
  scalers, encoders and selectors belong inside training folds and are replayed
  from the fitted model. A transformation has one owner, avoiding double
  application. Historical aggregates must respect observation availability.

## Source packaging

Simply nesting the folder would have captured Spark code in the model snapshot,
required group-package init files, and consumed the model's 64 KiB source budget.
Feature snapshot callers now explicitly exclude only the root `groups/` folder.
This covers single/branch training, competition and model-set scoring rules.
Generic package/composition capture retains every source file, including helpers
named `groups`; `custom/groups` also remains part of a model snapshot.

Static smoke still parses producer syntax. The feature job retains its existing
source hash, containment checks and bounded per-transform source read. Excluding
producers from model packaging does not remove those runtime checks. Do not import
producers from model recipes. Existing saved packages replay their stored source.
Old `src/feature_groups` configuration is rejected with the new canonical path;
the user had confirmed that no existing projects need migration.

## Verification

Relevant files were mapped through config validation, feature runtime/graph,
project source capture, ordinary/competition/set consumers and template docs.

Five focused regressions failed before implementation: new-path admission,
ordinary package source budget, static package smoke, competition capture and
combined model-set rule capture. An initial test-directory setup issue was
corrected before recording the four packaging failures.

After implementation, one deduplicated affected run passed **146 tests**:

```powershell
.venv\Scripts\python.exe -m pytest `
  skyulf-core/tests/integration/platforms/test_feature_group_config.py `
  skyulf-core/tests/integration/platforms/test_feature_group_runtime.py `
  skyulf-core/tests/integration/platforms/test_feature_group_graph.py `
  skyulf-core/tests/integration/platforms/test_feature_group_notebook.py `
  skyulf-core/tests/integration/platforms/test_databricks_project_package.py `
  skyulf-core/tests/integration/platforms/test_project_checks.py `
  skyulf-core/tests/integration/platforms/test_competition_project.py `
  skyulf-core/tests/integration/platforms/test_model_set_project.py `
  -q --tb=short -o addopts= -p no:cacheprovider `
  --basetemp=tmp_repro_artifacts/task183/affected
```

This includes actual fit/reload in fresh interpreters with producer files present,
fold-specific custom preprocessing, nested helper preservation and real local
MLflow model-set packaging. Three fixture warnings concern small-sample R2 and
MLflow input examples; there were no failing tests.

**3 additional tests passed** in `test_feature_bundle_generation.py`, with
`SKYULF_BUNDLE_OFFLINE_CLI=1` and `SKYULF_BUNDLE_CLI_TEST_PROFILE=offline`.
Real CLI generation, project smoke with group files, and strict target resolution
passed for single, competition and multi-target projects. The sandbox initially
blocked launching the installed CLI; the same localhost-only test succeeded with
execution permission. No workspace connection or cloud resource changes occurred.

Both tests in `test_feature_group_delta.py` collected successfully after updating
their fixture paths; real Delta execution was not repeated for this path change.
This is collection evidence only. Native Databricks acceptance remains deferred
by the user and SM-21a status is unchanged.

Final full Ruff lint, Ruff format (1579 files), full CI Ty scope, backend/core
Lizard CCN 10, template schema consistency and `git diff --check` passed.
Only formatting followed the affected test run; production library code remained
frozen through model artifact verification. No frontend or dependency changes.
