# Upgrade a generated Databricks project

## When you need it

Upgrade the Core pin, adopt a newer template graph, or add notebooks that were omitted during
original setup. Compare a freshly initialized project in a separate directory; regeneration is not
an in-place upgrade.

Public documentation describes the current revision. Read **Upgrade steps for generated Databricks
projects** for every intervening release in the
[0.9.x changelog](https://github.com/flyingriverhorse/Skyulf/blob/master/changelog/0.9.x.md). Keep
the project pin and release-specific contracts together.

## What to edit

Keep bundle identity, target resource keys, workspace root, output/state paths and edited recipes
stable. Transfer reviewed changes to notebooks, dependencies and graphs as one compatible set.

### Configuration validation and migration

New projects declare `config_version: 1` and `task`. The notebook validates
the resolved configuration before training, registry access or output writes:
resource names, distinct columns, reserved metadata names, budgets, model/task
compatibility, metric/threshold domains and action-specific snapshot inputs.
Actual source schema, keys and artifact compatibility are checked by the
existing data-read and scoring services against the real data.

`python src/tools/preview.py --action train` (or `--action score`) runs the same
check locally, without Spark or registry access, for every layout. For another
action in a `single_model` project, run this from the project root:

```python
from pathlib import Path

from skyulf.integrations.databricks.lifecycle.workflow import resolve_target_config
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config
from skyulf.integrations.databricks.projects.yaml_config import (
    project_config_path,
    read_workflow_config,
)

root = Path.cwd()
config = read_workflow_config(project_config_path(root))
loaded = load_project_workflow(config, root / "src/features")
resolved = resolve_target_config(loaded, {
    "catalog": "workspace",
    "input_schema": "my_inputs",
    "output_schema": "my_outputs",
    "metadata_schema": "my_models",
    "resource_suffix": "",
})
validate_workflow_config(resolved, action="approve")  # Or train, score, etc.
```

`load_project_workflow` adds the recipes and code from `src/features`; validating
the raw YAML without it fails with `pipeline.modeling.type must identify a
registered Core model`. Competition and multi_target projects load their
candidates or branches differently; use `preview.py` for them.

For an older project's loaded JSON, migrate explicitly:

```python
from skyulf.integrations.databricks.projects.workflow_config import migrate_workflow_config

updated = migrate_workflow_config(
    config, task="regression", score_handoff="after_alias_change"
)
```

Review/save `updated`, regenerate the affected job definitions and entrypoints, then
validate and redeploy together. Migration returns a copy; it does not upload
files, invent training dates, or change aliases. Legacy `auto_champion` maps
to champion/automatic; legacy `pinned_version` maps to pinned/manual approval.
Mixed policies, unknown config versions and contradictory existing task or
handoff settings are rejected. The notebook checks the deployed
`workflow_contract` and `deployed_score_handoff` markers against the JSON.
These detect stale generated definitions; they are not workspace permissions
or a complete audit of manually edited Jobs settings.

`test`, `syst` and `prod` have placeholder workspace and catalog bindings. Optional personal targets
are `test_development`, `syst_development` and `prod_development`; there is no standalone `dev`
target. Edit each target's host, catalog and input/output/metadata schema bindings, then validate
with its designated profile. Replace the placeholders with your approved workspace values.
Policy-backed compute uses the policy name, runtime, node type and cost tag supplied at
initialization. Serverless compute needs none of those fields.

For a company policy requiring `PayingRegNo`, select that value as
`cost_tag_key` and supply the approved registration number as `cost_tag_value`.
The `paying-reg-no-init.example.json` example demonstrates both settings;
the generic default remains `CostCenter`. This tag applies to policy-cluster
compute. Selecting it does not assign a model risk category.

Advanced configuration accepts an optional `risk_category`, such as `Low`,
`Medium`, or `High`. Leave it blank to omit the tag. The value remains editable
in `config/training.yml` defaults and is recorded on future training runs and model
versions; it does not change promotion gates or relabel existing versions.

## The command

```powershell
uv pip install -r deployment/train-requirements.txt
python src/tools/smoke.py
python src/tools/preview.py --action train
python src/tools/preview.py --action score
databricks bundle validate --strict -t test --profile YOUR_PROFILE
```

These checks precede deployment. Verify saved-artifact runtime compatibility and the
[monitoring store upgrade](databricks_monitoring_setup.md) before redeploying in your target.

## What you should see

Imports do not create a Spark session or cloud resource.

Regenerate and deploy notebooks, graph and the matching Core package together for graph contract 3.
Existing graph-2 bundles retain their legacy runtime path; changing only a
contract marker does not migrate a bundle.
The workflow configuration schema remains version 1.

Notebook entrypoints are explicit: generated `src/jobs/score.py` calls
`job_runtime.run_score_notebook`; lifecycle notebooks call
`job_runtime.run_lifecycle_notebook` with a fixed phase. Deploy new entrypoints
with the matching Core package. Existing direct `run_notebook(task_role=...)` callers
remain supported: score delegates to the score entrypoint, while lifecycle keeps
its sequential behavior and does not acquire durable phase/retry semantics.
`run_bundle_action`, `run_action` and `train_candidate` retain their APIs.

Edit recipe lists in `config/preprocessing.yml`, custom calculations in
`src/features/preprocessing.py`, model settings in `config/training.yml`, and
scoring/publication settings in `config/inference.yml`.
These remain project choices; common workflow fixes ship in the Skyulf wheel
instead of requiring edits to every generated notebook.

The library's `skyulf.integrations.databricks` modules are grouped under `jobs`,
`training`, `scoring`, `observability`, `model_sets`, `projects`, `data` and
`lifecycle`, with cross-domain contracts in `shared`. Crowded groups have smaller
subpackages: `jobs/{training,monitoring,lifecycle,shared}`,
`training/{fitting,competition,tuning,thresholds,weights,shared}` and
`scoring/{batch,incremental,shared}`. Monitoring lives in
`observability/monitoring`, with `spark`, `performance` and `serving` subpackages;
charts and reports have their own folders. For example, notebook adapters import
`skyulf.integrations.databricks.jobs.shared.job_runtime`.
Use these paths directly. The old `local_*` integration modules and renamed
APIs have been removed; update existing imports and model packages as described
in [upgrade integration code](#upgrade-integration-code).
The generated project's `src/jobs` directory remains unchanged.

The related MLflow library follows the same organization:
`skyulf.integrations.mlflow.{models,spark,lifecycle,registration,runs,shared}`.
Use `models.pipeline_model` and `models.feature_model` for fitted-pipeline
adapters. Package import leaves MLflow optional; pyfunc adapters load it when
selected.
Worker source snapshots and certificates continue to cover the entire Skyulf
package after the files move.

## Changes to adopt from earlier releases

### Selection and promotion policy fields

Earlier projects used one `model_selection_mode`. Generated projects use
`score_model_selection` and `promotion_policy` together, plus `score_handoff`.
Remove `model_selection_mode` when adding the two fields; mixed or partial
configurations fail before training or scoring. Migrate the configuration, all
notebook entrypoints and the job graph together; changing only a configuration
file is not enough. Direct Core callers can still pass the legacy value with a
deprecation warning; `migrate_workflow_config` above shows how each value maps.

### Final report links

The **Where to look** table needs two additions in a project generated before
they existed:

1. In `resources/train.job.yml`, add `workspace_host: ${workspace.host}` to the
   `base_parameters` of the `training_report` task. Generated files carry it in
   both layout branches. Without it, the table renders but every link is plain
   text.
2. In `src/jobs/training_report.py` (or `src/jobs/models_report.py` for
   multi_target), pass `display_html=globals().get("displayHTML")` to the
   runtime call. Without it, the task prints a plain text summary and no table.

Compare both files with a freshly generated project, then validate and redeploy.

### Replacing a champion produced by the previous release

Models load only under their producing runtime, so the first training after an
upgrade cannot compare its candidate with the old champion. Review the candidate
and approve it with `allow_unloadable_champion=true`; see
[champion from an older Core release](databricks_lifecycle.md#champion-from-an-older-core-release).
Until then, scoring the old champion needs the old runtime. Model sets must be
replaced with the release that produced their champion set.

### Lifecycle tags and alias recovery

Run every job that reads or writes lifecycle tags for one registered model on
the same Core version, and upgrade them together.

- `promotion_<event_id>` tags carry an `at` timestamp. Older Core readers reject
  it, so downgrading a job after such a tag exists makes its lifecycle checks
  fail. Current readers still accept tags without `at`; old tags are not
  backfilled.
- Alias recovery intent is stored as framed chunks with
  `"encoding":"json-framed-v1"`. Current readers also read the earlier unframed
  intent. An older reader cannot recover a framed intent and reports
  `Alias intent digest differs from its stored chunks`. Upgrade the operator
  environment that runs recovery as well.

### Model sets

- Sets registered before quality evidence was saved remain scoreable. Retrain and
  package them before approving through the Bundle. The low-level SDK keeps an
  explicit functional-only approval for those packages.
- `model_set_<branch>_*` version tags exist only on sets registered after they
  were introduced; older versions are not backfilled.
- Projects configured through `src/modeling/model_set.py`, a `src/composition/`
  package or Python recipe builders remain readable through their original
  loader. Do not define the same set or recipe in both Python and YAML.
- New tuning and fitted-configuration parameters appear only on training runs
  made with the updated Core; existing runs are not backfilled.

### Spark scoring

A whole-frame MLflow package logged without a Spark safety certificate cannot
use `inference_mode: spark`. Log a new certified version. Configurations without
`inference_mode` keep whole-frame scoring.

### Project layout and configuration files

- Projects configured through older Python or JSON files remain readable. Keep one
  owner per setting; do not define it in both the old file and YAML.
- To adopt the current layout, initialize a separate project, transfer values into
  its YAML files and copy custom functions into `src/features/`. Model versions
  already registered keep their saved source snapshot.
- `resources/workflow.jobs.yml` became `train.job.yml` and `score.job.yml`. Keep
  the `train`/`score` resource keys, bundle name and root so jobs are updated in place.
- The standalone `dev` target became test/syst/prod development pairs with their
  own roots and personal schema bindings. Generate the new layout separately, keep
  the old root during the change and never delete old jobs or tables as part of it.
- Answer files use plain-text key/input fields and task-specific
  `regression_*`/`classification_*` model and metric fields. Regenerate old answer
  files; their older names are still accepted.
- `max_bytes` became `max_input_mb`, and earlier key/result-time names became
  `record_key_columns` and `result_available_at_column`. Neither has an alias:
  regenerate and retrain.
- Temporal configurations must name their window mode explicitly.
- New projects have no `src/features/custom/` package. Scoring callbacks live at the
  end of `scoring.py` (`scoring.<function>`). Existing projects may keep `custom/`;
  `custom.<file>.<factory>` references still resolve.
- Training evidence from experimental releases, and candidates saved without
  sampling evidence, must be recreated before approval.

### Monitoring and serving

- New projects ask for the monitoring catalog and schema and start with
  `monitoring_enabled: "false"` when they are blank. Existing projects keep
  `monitoring_enabled: "true"` in `deployment/variables.yml`; with a blank
  destination their train and score jobs stop with
  `Set both monitoring_catalog and monitoring_schema.` Set both names, or set
  `monitoring_enabled: "false"`, then redeploy.
- Older monitoring stores need the owner to rerun store initialization, which adds
  `retraining_requests` and serializable isolation, before producers without
  CREATE privileges upgrade.
- Deploy train, score and monitoring jobs with the same Core so the
  development-target monitoring guard and activation-order metadata apply together.
- The score graph runs `prepare_monitoring -> monitoring_ready -> monitor_model`.
  The asynchronous `monitoring_invocation_id` and the `drift_report.py` /
  `retrain_on_drift.py` wrappers were removed; update projects that still call them.
- Upgrade every inventory reader before the first serving enrollment; older
  runtimes reject the serving field. Serving latency classes and logging-error
  counts appear only in reports recorded after the upgrade.
- Rollout `health.maximum_latency_ms` limits p99 execution time when it is recorded,
  with the maximum as fallback; earlier controllers checked the maximum. Review
  configured limits before upgrading the rollout job.
- Add `bundle_target: ${bundle.target}` to the `daily_rollout` task parameters so
  rollout evidence names the target that ran it instead of `monitoring.environment`.

### Upgrade integration code

Use the purpose-based module, class and function names from the
[Python SDK guide](databricks_sdk.md). The previous fitted-pipeline and
integration names containing `local` have been removed, including their
forwarding modules. Update those imports and call sites.

Replace `runtime="local"` with `runtime="standalone"`; `local` is not a valid
runtime value. Choose `engine="pandas"` or `"polars"` separately, and use
`runtime="databricks"` for UC Delta publication.

Saved pickle or MLflow packages that reference removed module or class names may
fail to load. Rebuild and log those model packages with the updated imports
before upgrading their consumers.

Persisted values such as `kind="local_pipeline"`, `mode="local_pipeline"` and
`execution_scope="whole_frame_local"` are unchanged. Keep these artifact-format
values; Python naming does not change execution admission, payload checks or
resource budgets. Portable bundles continue to use `predict_local`.

## Common errors and fixes

| Symptom | Fix |
| --- | --- |
| `Alias intent digest differs from its stored chunks` | The recovery environment runs an older Core; install the writers' version. It does not mean the evidence is corrupt. |
| Report links are plain text | Add `workspace_host: ${workspace.host}` to the `training_report` task. |
| Graph contract mismatch | Deploy matching notebooks, graph and Core; changing a marker alone is not migration. |
| A notebook is absent after toggling a feature | Copy that feature's generated notebook, resource and dependency changes from the comparison project. |
| Saved model rejects its runtime | Use the producer-compatible environment or retrain; changing a requirements pin does not migrate artifacts. |
| Monitoring view is missing a new column | Run the documented store upgrade before refreshing the dashboard; refresh does not change SQL definitions. |

The validation snippet loads the project's own files. The migration snippet
assumes an already loaded `config`. Both return data to review and do not
perform a cloud deployment.
