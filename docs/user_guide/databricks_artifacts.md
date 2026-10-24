# Read MLflow experiment artifacts

## When you need it

Use this reference when an experiment run's **Artifacts** tab contains
`lifecycle/`, `model/` and the training JSON files below. It explains what each
file records, which question it answers and which files must be read together.
These are saved results and evidence from that run, not settings to edit for
the next run. Change the project's configuration and train again for new settings.

## Find the model-producing run

Open **Jobs > training run > training_report > Output > Where to look > Model / run**,
then that MLflow run's **Artifacts** tab. You can also follow a registered model
version's source run. In a competition, each candidate has a fitting run; for a
model set, inspect the relevant component. The orchestration parent, a Charts
child and an approve/rollback run can have different artifact sets.

See [training reports](databricks_training_reports.md) for the navigation and
[MLflow Inputs](mlflow_tracking.md#training-datasets-in-inputs) for the separate
dataset entries. The filenames here describe Skyulf's candidate-training paths;
manual MLflow logging, other layouts and older releases can differ.

## Choose the artifact for your question

| Question | Start here |
| --- | --- |
| Which source table, version and split did this model use? | `training_data.json`, then `candidate_training_spec.json` |
| Which parameters were actually used by the fitted estimator? | `training_parameters.json`; add `tuning.json` for search history |
| Why did rows disappear before fitting? | `pre_split_filters.json`, then `training_filter_evidence.json` |
| Are these the same holdout/source values as the original run? | `holdout_membership.json`, `training_filter_evidence.json`, `monitoring_source_evidence.json` |
| Why was a candidate eligible or held back? | `candidate_comparison.json` and `quality_gates.json` |
| Did approval or promotion actually happen? | Final report, lifecycle decision/operator result and registry state |
| Where are the learned preprocessing and estimator? | The saved model package referenced by its model URI |
| Which request was used before training failed? | `training_snapshot.json`, `training_pipeline_config.json` and the failed task's output |

## Training-run JSON files

| File | What it contains | How to use it |
| --- | --- | --- |
| `candidate_comparison.json` | Candidate/champion versions and digests, dataset identity, holdout metrics, metric direction, improvement, thresholds, `eligible` and `reason`; sometimes a saved holdout reference. | Explain the recorded comparison. This is not a receipt that the champion alias changed. |
| `candidate_training_spec.json` | The candidate's final data-selection specification: source/version, keys, inputs/target, engine, split/window, label availability, sampling, weights, pre-split steps and enriched membership/evidence hashes. | Inspect the exact data policy and evidence used for later approval or monitoring. It is not the model hyperparameter dictionary. |
| `dataset_inputs.json` | Provenance joining source identities, feature names, training spec, filter evidence and source-content receipt; optional sampling, feature lookup, labels and conversion details. | Explain the native MLflow training/holdout **Inputs** entries, whose source metadata references this file and its digest. It does not upload another copy of every input row. |
| `holdout_membership.json` | `dataset_id`, `record_key_columns`, `holdout_rows` and `holdout_key_sha256`. | Check holdout identity and size. It stores key-column names and a fingerprint, not the individual keys or labels. |
| `monitoring_source_evidence.json` | `format`, dataset identity, column names, row count and `content_sha256` for the normalized source values actually read. | Prepared Spark monitoring references use this evidence to verify the original source content. Matching keys alone do not prove matching feature values. |
| `pipeline_config.json` | The effective Core pipeline recipe prepared before fitting: normalized search settings, projected fixed feature steps and applicable threshold/admission context. | Inspect how the requested recipe was prepared for training. It is not the complete final fitted state. |
| `pre_split_filters.json` | `requested_steps`, per-step `counts`, `source_rows`, `eligible_rows` and `excluded_rows`. Each count names the step/transformer and input/excluded/output rows. | Read fixed filtering before the final split. This does not describe all row exclusions during later holdout scoring. |
| `quality_gates.json` | A `gates` list with each rule's `metric`, `direction`, `threshold`, `value`, `passed` and `reason`; coverage rules appear when applicable. | Identify a failed quality or coverage check. All gates passing does not prove improvement requirements or the alias update succeeded. |
| `training_data.json` | A readable context summary: dataset identity, train/test source names and versions, split strategy, model type, engine, candidate date and optional time/risk/conversion metadata. | Quickly identify where the run's data came from. Despite the name, it does not contain training rows. |
| `training_filter_evidence.json` | Recipe/project-source hashes, key-column names, fingerprints and counts for source/pre-filter/surviving/train/holdout/sample populations, ordered filter transitions and optional group/weight evidence. | Trace how the training population was derived. Approval validates this evidence's identity, recipe, membership and counts when bound by the saved spec. |
| `training_parameters.json` | Actual fitted estimator constructor parameters, model type, recipe/step names, active split settings and ensemble details; optional weight/class-weight summaries. | Read full parameter summaries when MLflow's Parameters view shows a shortened value. Learned coefficients, trees and fitted transformer state live in the model package. |
| `training_pipeline_config.json` | The original loaded Core pipeline recipe supplied to candidate training, before deriving the effective recipe. It is a configuration dictionary, not a raw copy of the project YAML. | Inspect the original request. The documented SDK replay pairs this file with `training_snapshot.json`. |
| `training_snapshot.json` | An early training-spec snapshot: pinned source and selection settings, input budgets, split/window/sampling, label/feature bindings and engine. | Reconstruct the initial request. It is written before the current fit enriches membership/evidence hashes; it is not a table backup. |
| `tuning.json` | Search recipe, requested/effective metric, score direction, `best_score`, `best_params`, `n_trials`, trials and applicable nested-CV/threshold evidence. | Explain the search and chosen settings. A fixed-parameter model normally has no tuning artifact. Search scores are not the final holdout comparison. |

### Original recipe, effective recipe and fitted state

These three answer different questions:

1. `training_pipeline_config.json`: **What recipe was supplied?**
2. `pipeline_config.json`: **What recipe was derived before the data read and fit?**
3. Model package: **What transformations and model state were actually fitted?**

The effective recipe can include fixed transformations projected from pre-split
steps. Later fitting adds observed conversion/weight metadata and admission
evidence to the saved model. Therefore, neither configuration JSON is a substitute
for the fitted artifact.

For [the documented SDK replay](databricks_bundle_walkthrough.md), use the
original `training_pipeline_config.json` with `training_snapshot.json`. Feeding
the effective recipe back through preparation can duplicate projected steps.
This pair is not a generic command to resume a partially completed lifecycle job.

### Early selection versus final training evidence

`training_snapshot.json` is written early, before the current execution has
computed its final membership and filter evidence. `candidate_training_spec.json`
records the enriched specification after fitting. Their overlap is intentional;
the latter binds the actual candidate to its saved evidence.

Source counts describe the population materialized for that run, which may
already have a window, sampling or availability restrictions. They need not
equal the physical Delta table's total row count. Likewise, do not assume
`source_rows = eligible_rows + excluded_rows`: other selection steps may apply.

Digests fingerprint different things: holdout membership, source content, recipe,
model bytes or an entire evidence document. They are not interchangeable, and a
digest cannot restore missing rows. A document's `format: 1` or schema `version: 1`
is also distinct from the selected Delta table version in the training spec.
See [data selection and holdout](databricks_training_data.md) and
[monitoring enrollment](databricks_monitoring_enrollment.md).

### Search, quality and final promotion

`training_parameters.json` summarizes the actual fitted estimator's settings,
including defaults and selected tuned values. It does not serialize fitted state;
non-JSON constructor objects can be represented by their Python type.

In `tuning.json`, `best_params` describes the final search on training rows;
nested outer-fold selections remain under `nested_cv`. Check `score_direction`:
a negative-loss scorer can have a negative best score. Do not read it as a
positive holdout RMSE. Failed trials can have `score: null` and `status: failed`.
See [search strategies](databricks_tuning.md).

The saved comparison and gate files precede the final promotion decision:

- `eligible: true` does not mean the model became champion.
- `eligible: false` with `reason: no_champion` can be normal for a first model.
  First-champion initialization has its own threshold and quality checks.
- A gate's unavailable metric can have `value: null`; read its failure reason.
- Later approval uses the saved comparison/spec and checks their identities;
  today's edited YAML does not replace the candidate's saved evidence.

Read the final report, the relevant lifecycle result and registry state to
establish the outcome. See [approval and promotion](databricks_lifecycle.md).

## Inside the model artifact

`model/`, when shown under the run, is the packaged fitted model used for
prediction. The ordinary Skyulf MLflow package includes the pyfunc definition,
signature/input example, dependency declarations and a `local_pipeline` artifact.
The inner Skyulf artifact contains `manifest.json`, `pipeline.pkl` and, when
custom source was captured, `preprocessing.py`.

The package preserves learned preprocessing, the estimator and the saved
scoring/threshold behavior. It is more than model hyperparameters. A feature-lookup
model can use a Feature Engineering wrapper, so its directory tree can differ.
Use the actual model URI reported by the run or registered version instead of
constructing a path from this example directory name.

A successfully logged model does not by itself establish registration, approval
or live serving. Those are separate results. Model loading also uses the saved
runtime and integrity contracts; reading JSON metadata alone does not prove a
package loads and predicts successfully.

## Inside the lifecycle artifacts

`lifecycle/` contains durable workflow requests, intermediate data and stage
results. Files depend on the graph and the stages reached; SDK-only training
does not necessarily produce the Bundle's lifecycle records.

| Path below `lifecycle/` | Meaning |
| --- | --- |
| `request.json` | Run/job identity, action, pinned config/spec/champion and operator options. |
| `prepare.json` | Completed initialization record; the visible initialization task does not imply a file named `initialize.json`. |
| `load_data.json`, `prepare_dataset.json` | Source-read and fixed-filter/split results, including counts and references to saved data. |
| `data/source.parquet`, `data/train.parquet`, `data/holdout.parquet` | Actual intermediate rows. The split datasets precede learned preprocessing; these are not just hashes. Feature-lookup comparison replay can also save enriched holdout rows here. |
| `train.json`, `select_best_model.json` | Fitted-model evidence or the verified selected competition candidate. |
| `initial_evaluation.json` | Holdout metrics and dataset identity before model registration. |
| `registration.json`, `evaluate_register.json` | Concrete model registration receipt and the surrounding stage's completion record. |
| `compare.json` | Candidate result and comparison evidence. |
| `decide.json` | Single-model/competition decision, promotion policy and any alias-change result. |
| `generate_charts.json` | Chart task result, possibly `disabled`; not an alias decision. |
| `finalize.json`, `result.json` | Lifecycle finalization and the published workflow result. A finished run alone does not mean promotion or downstream scoring succeeded. |
| `operator.json` | Approve/reject/rollback result, potentially in a separate operator run. |
| `candidate_<name>.json` | Parent-run completion records for competition candidates. |
| `branch_<name>.json`, `register_model_set.json`, `evaluate_model_set.json`, `model_decision.json` | Model-set parent graph records; inspect component runs for their fitted-model evidence. |

Stage-completion records commonly contain `version`, `run_id`, `request_sha256`,
`phase`, `predecessor` and `output`. Not every file uses that envelope:
`request.json`, `registration.json` and `initial_evaluation.json` have distinct
formats. The runtime also validates completion tags and digests; merely finding
a JSON file does not prove a stage completed.

The top-level JSON summaries do not make the whole run metadata-only: these
Parquet artifacts contain real rows, and model state/input examples can contain
data-derived values. Apply the appropriate experiment access and retention to
the complete artifact set.

## When an expected artifact is missing

Check the run, layout, saved options and first failed task before interpreting
absence:

- A failed fit can leave the early config/snapshot files without a model or
  post-fit summaries.
- Fixed-parameter training need not create `tuning.json`. Enabled features may
  add `cross_validation.json`, `training_selection.json`, `decision_threshold.json`,
  `preprocessing_probe.json`, chart reports or other evidence.
- A graph or SDK path may not execute the stage that writes a lifecycle record.
  Older runs are not retroactively filled with newly introduced artifacts.
- Required candidate approval evidence being absent is not a passing check.
  An uncertain registration/alias operation is not proven unsuccessful just
  because its final receipt is missing; inspect its recorded state.

These training artifacts do not establish batch-scoring CDF progress or a
prediction-table commit. Use [scoring receipts](databricks_scoring.md) for that,
and [troubleshooting](databricks_troubleshooting.md) for a failed task.
