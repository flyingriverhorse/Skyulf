# Using the Databricks Bundle

Use this walkthrough to generate a project, train a candidate, inspect its reports,
activate a model, and run scoring. You need an authenticated Databricks CLI profile,
access to your source tables and model registry, and the matching Skyulf wheel.
See the [Bundle configuration guide](databricks_bundle.md) for detailed settings.

Training uses bounded pandas or Polars data on job compute. Whole-frame scoring
uses the saved engine; `inference_mode="spark"` instead runs admitted pandas
models on distributed workers. Neither choice changes which data training may use.

## Prepare the project

From the Skyulf checkout:

```powershell
databricks bundle init skyulf-core/templates/databricks --output-dir ./generated
```

Choose the project directory created under `generated` and work from there.
Review these files before deployment:

| File | Configure |
| --- | --- |
| `config/training.yml` | Shared `defaults` and named `models`: source, keys, features, target, split, CV, estimator, weights and quality gates |
| `config/inference.yml` | Scoring source, model selection, output destination and inference mode; `model_set` controls multi-target publication |
| `config/pre_split.yml` | Ordered fixed training-eligibility recipes |
| `config/preprocessing.yml` | Ordered fitted preprocessing recipes |
| `src/features/` | Custom Python transformations and scoring functions selected by configuration |
| `deployment/targets.yml`, `deployment/variables.yml` | Workspace/UC bindings, identities, compute and schedules |

`single_model` has one named model; `model_competition` compares candidates for
one target; `multi_target` trains independently named targets and packages a
coherent model set. Do not create a second `workflow.json` or Python model file
alongside generated YAML. Multiple configuration owners fail validation.

Install the matching wheel in your preview environment and place the deployment
wheel in `dist/`. Preview the generated project without submitting a job:

```powershell
python src/tools/preview.py
python src/tools/preview.py --action train
```

Preview uses the development bindings by default; supply the corresponding
catalog/schema/suffix arguments for another target. It loads trusted project
code and resolves configuration, but does not fit a model or read source rows.
Then validate and deploy from the generated directory:

```powershell
databricks bundle validate --strict -t dev --profile <profile>
databricks bundle deploy -t dev --profile <profile>
```

Choose manual schedules while learning the workflow. A scheduled job can start
after deployment when its pause status is `UNPAUSED`.

## Choose the preprocessing phase

Define named lists under `recipes` in `config/pre_split.yml` and
`config/preprocessing.yml`. Select their names with `pre_split_recipe` and
`preprocessing_recipe` in training defaults or a model entry. Custom functions
live in the corresponding `src/features/pre_split.py` or `preprocessing.py`.

| Operation | Phase | Inference behavior |
| --- | --- | --- |
| Fixed eligibility rules | Pre-split | Select training/evaluation rows; scoring eligibility is separately configured |
| Fixed normalization needed by eligibility | Pre-split | Saved feature normalization is applied once to raw scoring input |
| Imputation, scaling, encoding, feature selection | Preprocessing | Reuse fitted training state; each CV fold fits its own state |
| Oversampling/undersampling | Preprocessing | Training rows only; no resampling of holdout or scoring input |
| Lag or rolling | Preprocessing with ordered context | Use the supplied frame or explicit carry history |
| Dataset profiling/snapshots | Training diagnostics | Prediction passes through without rebuilding the training report |

For example, normalize a sentinel before testing training eligibility:

```yaml
version: 1
recipes:
  known_income:
    - name: income_sentinel
      transformer: ValueReplacement
      params: {columns: [income], to_replace: -999, value: null}
    - name: known_income
      transformer: DropMissingRows
      params: {subset: [income]}
```

Place this in `config/pre_split.yml` and select `pre_split_recipe: known_income`.
Do not repeat the same normalization in preprocessing. Scoring turns the sentinel
into missing data without dropping the requested row; add a fitted imputer if
the estimator should accept it. Keep learned statistics after splitting.

### Deduplication and your own training filter

Deduplicate with explicit columns and a keep policy. Source record keys must
still be unique; deduplication is not a substitute for group-isolated splitting.
Custom pre-split filters declare fixed eligibility and must preserve retained
rows, order and labels. See [custom preprocessing recipes](databricks_bundle.md#custom-preprocessing-recipes)
and the generated `PREPROCESSING.md` for function factories and restrictions.

## Training data flow

The job pins a Delta source version, selects the configured window and eligible
sample, applies fixed training filters, creates the final split, and then fits
preprocessing/model parameters inside training folds. Holdout rows stay outside
search and fitting. Label availability filtering is independent of observation
time and split policy.

Review `training_window_mode`, optional `training_sample_rows`, `split_strategy`
and `cv` separately. Random splitting needs no event dates. Temporal splitting
requires the configured time column and boundaries. Null `training_version`
resolves latest once at invocation; a concrete version pins that snapshot.
A rolling window changes at invocation, while a fixed window keeps its boundaries.

## Three independent decisions

| Setting | Location | Meaning |
| --- | --- | --- |
| `promotion_policy` | Training policy, or inference `model_set` policy | `manual_approval` waits for an operator; `automatic` applies saved quality gates |
| `score_model_selection` | `config/inference.yml` | Resolve the controlled champion or use a pinned `model_version` |
| `score_handoff` | `config/inference.yml` | `after_alias_change` starts scoring after a successful champion transition; `disabled` leaves scoring to its own trigger |

Manual approval does not imply manual scoring. A pinned scorer keeps using its
pinned version even after champion changes. Rejection and training without a
champion transition do not trigger score handoff.

## The two jobs and their tasks

The core lifecycle has `train` and `score` jobs. Optional feature engineering,
monitoring and dashboard resources depend on project configuration.

| Training task | What to inspect |
| --- | --- |
| `initialize_run`, `choose_action` | Resolved inputs and the requested train/approve/reject/rollback action |
| `load_data`, `prepare_dataset` | Pinned source and split evidence for single/competition layouts |
| `train_and_tune` or `train_<name>` | Actual fit, fold/search results and saved artifacts |
| `register_model` or `register_model_set` | Concrete registered candidate identity |
| `evaluate_model` or `evaluate_model_set` | Protected holdout metrics and comparison evidence |
| `model_decision` | Quality and lifecycle decision |
| `training_report` | Final status, report links and next actions |

In multi-target layouts, the graph's `training_report` task runs
`src/jobs/models_report.py`. The score job's `score` task runs the scoring path;
optional `recover_predictions` handles configured recovery. The final report and
committed receipt describe what actually happened.

## Train a candidate in the UI

1. Open **Workflows ? your train job ? Run now**.
2. Set `lifecycle_action=train` and review the resolved source/version, split,
   limits and model configuration.
3. Open the run graph and inspect the fitting task. A successful fit is not yet
   evidence that registration, quality evaluation or activation succeeded.
4. Open **training_report ? Output** for the final candidate and decision.

For competition, only the selected winner proceeds to the protected holdout and
champion comparison. Multi-target training retains a separate fitting run for
each branch; activation operates on the complete model set.

## Understanding the run settings form

Run parameters select a lifecycle action and supply its concrete identities.
The generated configuration owns model recipes and normal source/split settings.
Use the matching target bindings and leave inactive fields unset. Preview the
configuration before running when you change source dates, model settings or
resource budgets.

### Why the comparison still has a SHA-256 digest

The digest binds the reviewed candidate, dataset, metric policy and expected
champion. Approval checks that evidence instead of silently reevaluating a new
population with today's edited files. A stale champion or changed identity fails
explicitly; inspect the new state before deciding how to proceed.

## Reading the notebook result

The output contains human-readable sections and the complete JSON result for
automation. Check model/version, metrics, quality decisions, status and reasons.
Use the MLflow run link to inspect full artifacts, search/CV reports and saved
input evidence. An absent optional section means it was not produced; it is not
a successful check.

### Preprocessing diagnostics

In `config/training.yml`, opt in with:

```yaml
defaults:
  preprocessing_probe: true
```

After training, open **training_report ? Output ? Preprocessing diagnostics**,
or the individual fitting run's **MLflow ? Artifacts ? preprocessing_probe.json**.
Competition/multi-target models have their own fitting runs. The default is off.
The check uses a saved/reloaded artifact and up to the first 256 holdout rows,
with an 8 MiB input/output limit; it does not refit or change promotion gates.

`requires_context` means a group, window or global operation cannot be checked
as independent rows. Later steps can be `not_run`; empty input can be
`not_supported` even when ordinary sample checks pass. See the
[diagnostic guide](preprocessing_context.md) for all statuses and an executable
history example. A passing report is not worker or endpoint approval.

### Inspect failures and replay a pinned training input

Open the failed task and its MLflow run. `training_snapshot.json` records the
pinned source and input/split/budget settings; `training_pipeline_config.json`
records the original pipeline input and saved project source. Keep source Delta
history and the recorded dependencies available for replay.

A direct SDK replay reconstructs `TrainingSpec` from the snapshot, including
aware datetimes and `TrainingDateSpec` objects, and passes the original pipeline
configuration to `train_candidate`. Restore trusted saved project registration
for custom nodes. Do not substitute the effective `pipeline_config.json`, which
can already contain projected pre-split feature transformations. For a generated
project, restore the corresponding YAML and Python assets in a separate reviewable
copy; do not overwrite newer project source merely to inspect a failed run.

## Where to find `next_actions`

Open **Workflows ? train run ? training_report ? Output**. In single-model and
competition results, copy the `next_actions.approve`, `.reject` or `.rollback`
fields produced for that concrete result. Do not type a model alias where a
concrete version is requested.

For multi-target results, inspect `model_set_candidate.version` and
`quality.expected_champion_version`; use `none` when there is no champion.
Approval and rollback act on the complete set. Keep the promotion receipt from
the successful decision for any later rollback.

## Approve without training again

Run the existing **train** job with `lifecycle_action=approve`, the concrete
`candidate_version`, and `expected_champion_version` from the reviewed result.
This loads saved artifacts and evidence; it does not fit a new model. Both manual
and automatic activation enforce the saved quality gates and current expected
champion. The candidate must be the controlled nominated challenger.

On success, inspect the returned receipt and champion alias. If score handoff is
enabled, follow the child score run and inspect its publication separately.

## Reject a candidate

Run the train job with `lifecycle_action=reject`, the candidate and expected
champion versions, and a nonempty `rejection_reason`. Rejection records the
decision without training or scoring and leaves the version available for
inspection. Repeating the same request is checked against its saved evidence;
a changed reason or identity is not silently treated as the same request.

## Roll back a completed promotion

Copy `next_actions.rollback` from the successful promotion result. Supply
`lifecycle_action=rollback`, `promotion_receipt_json` and the expected current
champion. The operation validates the recorded transition before restoring the
prior champion or complete prior model set. Initial activation has no previous
champion to restore. An unrelated current challenger is preserved.

## What happens to existing predictions?

Changing a model alias does not itself rewrite a prediction table. Score resolves
one concrete model/set version and applies the configured change policy. Append
mode preserves older rows and their provenance; a configured full rebuild
recomputes the current snapshot and atomically replaces compatible output.
Schema changes need a compatible destination. Model changes involving carried
temporal history require rebuilding the corresponding history, not mixing states.

## Aliases and results to inspect

Inspect the concrete registered candidate, `champion`, `challenger`, preserved
previous aliases and promotion/rejection receipts. A registered version can exist
after a later failure without having become champion. A scoring result should
identify source watermark, model/version, row counts, no-op/rebuild outcome and
Delta commit receipt. Match outputs by record keys.

## Recovery and schedules

After a scoring failure, inspect the target receipt before retrying. The writer
can recognize an acknowledged or uncertain prior commit; deleting state or
inventing a new watermark can duplicate work. Recovery options are explicit and
do not convert arbitrary source updates/deletes into incremental inserts.

Train and score schedules are independent in deployment variables. A paused
schedule prevents clock triggers, not already queued or manually started runs.
Both jobs serialize their own active runs; protect registry and output ownership
from outside writers too. A successful no-op score is different from a queued job.

### Configure independent clocks and training data

Use training schedule settings for when to attempt fitting, scoring schedule
settings for when to inspect new input, and training-window settings for which
observations are eligible. The cron timezone does not define the source timestamp
timezone. Preview rolling boundaries and result availability cutoffs before
activating a schedule. See [source windows](databricks_bundle.md#source-windows-are-independent-of-splitting-and-scheduling)
for the detailed contract.
