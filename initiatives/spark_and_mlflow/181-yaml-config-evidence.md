# Optional single-source YAML configuration

Implemented locally on branch 093; no cloud execution, commit or push by this
configuration agent. The Bundle initializer still emits the existing Python/JSON
format. A generated migration tool provides the explicit YAML entrypoint; direct
YAML generation as the initializer's default is not delivered.

## Usage and ownership

Run `python src/tools/migrate_config.py` once from an initialized project's
environment. It derives its project root from the script, with no arguments.
Then run `python src/tools/smoke.py`, `python src/tools/preview.py` and
`python src/tools/refresh_training_graph.py`; validate and redeploy the Bundle.

The migration reads only recognized generated static declarations through AST;
it never executes their Python factories. Unknown/custom code, decorators,
annotations, executable argument defaults, duplicate literal keys and occupied
output/backup paths are rejected. It stages and validates YAML/model structure,
retains byte-exact originals in `.skyulf-yaml-backup`, publishes outputs and
removes the backed-up modeling declarations. A publication failure restores the
original files and removes its new YAML files. This is a local project migration,
not a concurrent-deployment transaction; do not edit the project during it.

New templates exclude backup and staging paths through the generated `.gitignore`.
When manually installing this migration tool into an older Bundle, also add
`.skyulf-yaml-backup/` and `.skyulf-yaml-stage-*/` there. Explicit optional sync
exclusions produced strict CLI warnings before those directories existed; the
final template uses Git ignore rules and one `src/**/*.py` include instead.

`config/training.yml` owns training defaults and model declarations;
`config/inference.yml` owns scoring controls and optional model-set settings.
Residual pipeline structure stays in `workflow.json`. Disjoint JSON/YAML settings
can coexist. Duplicate ownership is rejected even when values are identical;
move shared JSON fields into YAML `defaults` before overriding them per model.
Custom preprocessing, pre-split and scoring functions remain in Python.

Minimal model declarations can use:

```yaml
version: 1
defaults:
  task: regression
  model: {type: ridge_regression, params: {alpha: 1.0}}
models:
  revenue:
    target_column: revenue
  cost:
    target_column: cost
    model: {type: linear_regression}
```

This example assumes the project's remaining required source/registry/split
settings are supplied once in the shared defaults or residual JSON. For a single
model, declare exactly one model. Use `training_layout: multi_target` in defaults
for the example's two independent targets; `model_competition` instead means
multiple candidates sharing one task, target, holdout and CV configuration.

Model entries and defaults accept `model: {type, params}`, optional `tuning`
using existing tuner settings, `decision_threshold`, `explainability`, and named
`preprocessing_recipe`. Single and multi-target projects also accept
`pre_split_recipe`; `features_path` is restricted to multi-target declarations.
Whole nested mappings replace shared values; no implicit deep merge occurs.
Competition candidates may override only model, tuning, threshold, explainability
and preprocessing recipe. Shared competition defaults cannot select another
feature path or pre-split recipe.

```yaml
version: 1
inference_mode: spark
spark_udf_prediction_batch_rows: 1000
```

Inference YAML accepts existing inference mode/environment/batch, source/output,
model version/selection/change, CDF recovery and score-handoff fields. Optional
`model_set` uses the existing validated model-set structure including
`composition_config`; declarative composition rules are preserved unchanged.

All YAML has version 1, a 64 KiB source limit and a 32-level nesting limit. Parsing
rejects aliases, duplicate keys, non-string keys, nonfinite values and unknown
fields. Date strings need quotes. Model/tuning parameters retain their existing
runtime validators. Static smoke validates declarative models without executing
custom project Python; it cannot certify custom code or cloud/data contracts.

Migration removes only values equal to existing effective defaults: default CV
fields, default chart bounds, optional null source/lifecycle values, an unused
default sampling seed and default parsing maps without time columns. Selected
model parameters, populated search spaces, explicit weight settings and
nondefault values remain. Default generated training YAML measured 46 lines for
single, 152 for the three-candidate competition, 67 for two targets; competition
keeps independently populated model search spaces.

## Implementation and evidence

New modules: `projects/yaml_config.py`, `yaml_models.py`, `yaml_migration.py`.
Adapters: project/competition loading, notebook config, branch configs, model-set
settings and static smoke. Generated preview and graph refresh use the central
reader. The generated migration tool, YAML/feature-source sync patterns and a
direct shared PyYAML 6.0.3 runtime pin are included. The root agent owns final
dependency-lock resolution and repository-wide CI gates.

- New YAML tests: 27 passed in 1.90s at the final migration revision. Regression
  coverage includes saved fitted replay after editing YAML, competition/branches,
  model-set settings, duplicates, JSON/Python ownership, parser depth, custom
  factory refusal, rollback, inert static smoke and effective-default equivalence.
- Affected explicit 11-file union: 258 passed, 14 optional CLI skips, 5 existing
  warnings in 121.87s. Files: `test_databricks_yaml_config.py`,
  `test_project_checks.py`, `test_databricks_project_preprocessing.py`,
  `test_competition_project.py`, `test_model_set_project.py`,
  `test_databricks_branch_template.py`, `test_databricks_bundle_notebook.py`,
  `test_databricks_bundle_template.py`, `test_competition_template.py`,
  `test_databricks_tuning_template.py`, `test_databricks_workflow_preview.py`.
  The union preceded migration-only compaction and the final model-set literal
  guard; the final 27-test run covers those changes. No repeated legacy suite was
  needed.
- Scoped Ruff, Ruff format 16 files, Ty, and Lizard CCN 10 passed. `git diff --check`
  passed. A final narrow Bundle-template parse test passed after sync changes.
- Installed Databricks CLI rendered all three default layouts locally using
  isolated dummy localhost credentials. The existing `skyulf` OAuth refresh was
  expired; no account configuration was changed. CLI made no cloud deployment.
- All three real generated projects migrated and passed static smoke and runtime
  recipe resolution. Reconstructed original configurations and migrated ones were
  equal after normalizing effective omitted defaults and excluding only expected
  provenance-source changes. Graph refresh succeeded for single (no named tasks),
  competition (3 named candidates) and multi-target (2 named branches).

All Python checks used `.venv/Scripts/python.exe`, explicit test files and
`-o addopts=`; basetemps are under ignored
`tmp_repro_artifacts/task181/yaml/`. Original CLI fixture Bundles are in
`generated/{single_model,model_competition,multi_target}/yaml_generated` beneath
that directory. Compact equivalence fixtures are in `generated_compact/` and
contain config/source only. No native Databricks acceptance is claimed.
