# Score saved models and publish Delta predictions

The generated `score` job loads a saved model, finds eligible source records, produces predictions,
and commits them with source/model provenance. It does not train a model. Use this guide after
[training and selecting a version](databricks_bundle_walkthrough.md). For a complete project
sequence, start with the [Databricks walkthrough](databricks_bundle_walkthrough.md).

The example below scores customer churn from a separate unlabeled Delta table.
It shows exactly where to configure the source, version, eligibility and output.
Workspace SQL and job results are labeled **expected**; local configuration and
callback checks do not establish a deployed prediction run.

## 1. Choose the source and saved model

Keep these responsibilities separate:

| File | Scoring responsibility |
| --- | --- |
| `config/training.yml` | Model identity, record keys, feature columns, training recipe and local row/byte limits. |
| `config/inference.yml` | Scoring source, prediction destination, version selection and change/recovery policies. |
| `config/scoring.yml` | Rules captured with the next model version: reuse pre-split, eligibility, business outputs and combined rules. |
| `src/features/scoring.py` | Optional custom callbacks selected with `custom: scoring.<function>`. |
| `resources/score.job.yml`, `deployment/` | Deployed task graph, compute, schedule, identity, timeouts and retries. |

For the
[customer preprocessing example](preprocessing_recipes.md#complete-phase-files-for-a-small-customer-example),
use these **training defaults fields**, retaining the rest of the generated classification
configuration:

```yaml
record_key_columns: [customer_id]
input_columns: [age, income, tenure, segment]
target_column: churn
```

The record key must identify one source observation, be unique/nonmissing, and
remain outside model inputs. Use `BIGINT` or `STRING` for the generated
single-model prediction publisher. A date/timestamp is not a supported record
key; a historical customer table usually needs a separate observation ID.

Create the following **new** scoring source in Databricks, replacing the schema
with your own. The `churn` label is intentionally absent. Match these input types
to the training model's saved schema; changing the source names alone cannot
change a saved model's input contract. For this example, load the training
columns `age`, `income` and `tenure` as DOUBLE too, before fitting; keep
customer_id as BIGINT, segment as STRING and churn as BIGINT. This permits
nullable numeric scoring inputs under the same saved schema. If an existing
model was fitted with integer age/tenure, align the source to that contract or
fit a new version using this explicit schema. Preview does not inspect table
dtypes.

```sql
CREATE TABLE workspace.demo.customer_score (
  customer_id BIGINT,
  age DOUBLE,
  income DOUBLE,
  tenure DOUBLE,
  segment STRING
) USING DELTA TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');

INSERT INTO workspace.demo.customer_score VALUES
  (101, 35.0, 2500.0, 3.0, 'A'),
  (102, -1.0, 4000.0, 4.0, 'B'),
  (103, NULL, 1500.0, 2.0, 'A'),
  (104, 60.0, NULL, 5.0, 'NEW');
```

CDF must be enabled before incremental scoring starts. The source and output
must be different Delta objects. The job identity needs source read/CDF access,
model access and prediction-output write permissions. Do not prefill the output
with arbitrary predictions: first-run bootstrap requires a compatible empty
output or a new output provisioned by the workflow.

For a single model or competition winner, replace `config/inference.yml` with
this complete local-scoring example. Replace version `1` with the concrete
version you inspected in training evidence:

```yaml
version: 1
batch_compute: single_node
initial_scoring: full_snapshot
score_source_table: workspace.demo.customer_score
prediction_table: "{catalog}.{output_schema}.customer_scores{resource_suffix}"
score_model_selection: pinned_version
model_version: "1"
on_model_change: score_new_rows
rescore_when_history_expires: false
follow_champion: false
```

`local` means bounded whole-frame prediction, including on Databricks compute.
It is not a choice of workstation versus cloud. `max_rows` and `max_input_mb`
remain in training.yml and bound each local scoring materialization. The first
run can read the entire source, so budget for it as well as later increments.

| Selection | Single model / competition | Independent-target model set |
| --- | --- | --- |
| Normal run | `pinned_version` uses the positive string model_version; `champion` resolves the controlled champion for that run. | Uses the controlled champion of `model_set.model_name`; approve a set champion first. The ordinary top-level pinned selector is not the set selector. |
| One-run override | `score_model_version` selects a concrete model version. | The same parameter selects a concrete **set** version, not one component's version. |
| Training score handoff | After a successful alias change, uses the configured scoring selector. | After an approved set alias change, uses the set selection. |

Selecting a pinned version does not approve it, change aliases, or retrain it.
Changing to champion selection does not by itself make an unapproved candidate
the champion. A model set carries its saved component versions; do not try to
replace one component through the score job's version parameter.

Training and scoring reference the same existing input table by default; neither
reference creates it. Separate them when labeled training data and new scoring
data have different lifecycles. With the default `full_snapshot`, the first score includes all
existing rows, including historical labeled observations; confirm that population
before running. [Choose a table layout](databricks_training_data.md#choose-a-table-layout)
compares one shared table with separate measurement and outcome tables.

### Start with new records only

Set this in `config/inference.yml` **before the first score run** when existing
rows are training history and should not receive production predictions:

```yaml
initial_scoring: from_now
```

Quick and detailed setup both ask for this choice. The default is
`full_snapshot` for every layout. It applies to single models, competition
winners and model sets, using local pandas/Polars or supported Spark UDF
scoring. It does not change training selection or serving endpoint requests.

For example, suppose the source has 50,000 rows at Delta version 12:

| Run | Source | Result with `from_now` |
| --- | --- | --- |
| First score | Version 12, 50,000 old rows | Records version 12 in the prediction receipt; writes zero predictions. |
| Repeat without changes | Still version 12 | No-op; the checkpoint stays unchanged. |
| Next score | Version 13 inserts 100 new keys | Scores those 100 rows. The 50,000 old rows remain excluded. |
| Model changes with `rescore_all` | Current source snapshot | Recomputes eligible rows added after version 12; keeps the original exclusion boundary. |

The first score succeeds and can complete a training-to-scoring handoff even
with zero predictions. Monitoring has no production predictions to evaluate
yet; this is not a zero error rate or a measured performance score. An append
after the pinned version, including one arriving during the first run, belongs
to a later scoring run.

The receipt fixes `initial_scoring` and `scoring_start_version`. Changing the
choice on an existing prediction target fails; use a new target for a separate
scoring history. Existing projects and historical receipts continue to use
`full_snapshot` unless a new target explicitly selects `from_now`.

**Retention and keys:** record keys must be immutable and never reused.
Normal increments need only the CDF interval since the last successful score.
A normal model-change or model-set source-correction rebuild uses keys in the
last committed prediction snapshot (including excluded rows), plus new inserts
after its source watermark, joined to the current source snapshot. Keep that
prediction checkpoint and the recent CDF interval readable. It does not need
the original starting snapshot, even when months have passed.

Approved **CDF-expiry recovery** cannot reconstruct missing insert history this
way. It needs the source snapshot at the original starting version to exclude
its keys. Retain that snapshot and its data files if you require this recovery
option. If it has expired or been removed by VACUUM, recovery fails; it never
silently expands to all historical rows. Restore the snapshot or choose a new
prediction target and explicitly establish a new starting policy. Source
updates/deletes still follow the separate source-change policy.

## 2. Decide eligibility before training the version

Scoring rules are saved with the model. Configure them **before training and
packaging the version you will score**. Editing today's scoring.yml or scoring.py does not
rewrite an already registered version.

### The three scoring sections

`config/scoring.yml` has three rule sections. Each runs at a different moment:

| Section | Runs | Does | Example |
| --- | --- | --- | --- |
| `eligibility` | Before the model | Decides whether a row may get a prediction. | The hour is missing, so this record gets no prediction. |
| `outputs` | After the model | Adds columns computed from the prediction and inputs. | Probability 0.85 adds `risk_band: high`. |
| `combined` | After all models of a model set | Computes one result from several models. | Revenue prediction minus cost prediction gives profit. |

A row rejected by eligibility is not deleted. It stays in the result with
`scoring_status: excluded`, its `exclusion_reason` and an empty prediction.
Outputs run only on predicted rows and keep the raw prediction unchanged.

The [bike demand example](https://github.com/flyingriverhorse/Skyulf/tree/master/skyulf-core/templates/databricks/examples/real_data/bike_demand_set)
uses all three:

1. Eligibility: a row without `hr` is excluded with the reason `missing_hour`.
2. Outputs: each model adds `nonnegative_demand`. A raw prediction of -2 gives 0
   in the new column; the raw prediction is kept.
3. Combined: `total_demand` adds the `nonnegative_demand` columns of the `casual`
   and `registered` models. If either model excluded the row, no total is
   produced; a missing prediction is never counted as zero.

Two settings with similar names do different things:

- `mode: combined` applies the saved pre-split filters, then the eligibility
  rules, then the outputs.
- The `combined:` list defines cross-model rules. A single model does not use it.

Choose `mode: custom` to use only your own eligibility and output rules, and
`mode: pre_split` to reuse only the pre-split filters.

| `mode` in scoring.yml | What runs |
| --- | --- |
| `pre_split` (default) | Saved pre-split eligibility; empty recipe leaves the ordinary prediction schema. |
| `custom` | Custom eligibility, model prediction, then custom output rules. |
| `combined` | Saved pre-split eligibility, custom eligibility on its survivors, model prediction, custom outputs. |

With pre_split/combined, a filter reading the target needs an explicit decision:
`skip_target_pre_split_steps: true` skips those filters because scoring has no
actual label. The default false rejects that recipe, including during training
preview. It does not invent labels. A mixed filter reading both target and
features is skipped as a whole, so split the rules when the feature check should
remain. Custom mode rejects a true skip flag because it does not reuse pre-split.

Pre-split replay uses a copy to decide membership; the saved model applies its
fixed feature normalization once to the original accepted inputs. Combined
custom callbacks also receive those original accepted inputs. A filter's input
must be available in input_columns; it can later be dropped in preprocessing if
it is not a model feature. Deduplication compares the current batch, not all
records ever scored.

If pre-split excludes every row, neither custom callbacks nor the model runs.
An empty recipe in combined mode still runs its custom sections. The target-skip
switch does not change the target or synthesize a label; fixed target/feature
normalization is projected onto features for saved inference. Preview lists the
retained and skipped step names. Empty pre-split mode retains ordinary output
columns when there is no active scoring policy.

The generated `config/scoring.yml` starts with `version: 1`, `mode: pre_split`,
`skip_target_pre_split_steps: false` and empty `eligibility`, `outputs` and
`combined` lists. Changing mode alone does not activate sample rules. Nonempty
eligibility/output lists require custom or combined mode. `combined` is a
separate cross-model list, described in the [model set guide](databricks_model_sets.md#write-combined-rules).

### A complete YAML rule example

Replace `config/scoring.yml` with this document. It selects custom mode, so the
reasons come from these eligibility rules. No Python edits are needed:

<!-- example:customer-scoring -->
```yaml
version: 1
mode: custom
skip_target_pre_split_steps: false
eligibility:
  - name: observed_age
    check: required_values
    params: {columns: [age]}
  - name: adult_age
    check: value_range
    params: {column: age, minimum: 18.0, maximum: 120.0}
outputs:
  - name: churn_risk_band
    version: "1"
    rule: prediction_band
    params:
      column: probability_1
      thresholds: [0.3, 0.7]
      labels: [low, medium, high]
      output: risk_band
    columns: [{name: risk_band, dtype: string}]
combined: []
```

Each rule has a unique name and a string version (default `"1"`). Required-values
checks take `params.columns`; range checks take a column and at least one finite
inclusive minimum/maximum. Range checks reject missing, nonnumeric and infinite
values; required-values checks report missing fields in the declared order.
Bands require increasing finite thresholds and one more label than threshold.
Their single declared column must match params.output (default band), dtype string.
Unknown fields, duplicate YAML keys and invalid parameters fail during preparation.

For project-specific logic, replace check/rule with `custom: scoring.<function>`
and keep params and declared columns. Eligibility receives `(frame, params)`;
outputs receive `(frame, predictions, params)`. Implement the named helper in
`src/features/scoring.py`, with relative imports for adjacent project modules.
Custom references select captured project code, not external import paths or YAML
expressions. Per-component eligibility/output callbacks receive only that model's
saved input columns; shared callbacks must read columns available to every
component. Cross-model combined callbacks instead receive the set's union of
component input columns and keys, plus the component prediction results.

Older Python-only projects still use exported `build_model_rules()` or legacy
`build_scoring()` (the former takes precedence), plus `build_combined_rules()`.
A project containing scoring.yml must not also define/export those builders:
remove them after moving declarations to YAML, retaining custom helper functions.
Previously saved model versions continue to replay their saved policy and code.

The probability rule assumes the saved class order is `[0, 1]` and the classifier
exports probability_1 for churn. Check the saved output/class schema; the suffix
is a class position, not a promise about every target's label. If the estimator
has no probability output, choose an available output or remove this rule.
The bands are business thresholds, separate from the model's decision threshold.

Eligibility rules return one reason per row: null means accepted, text means
excluded. They do not drop rows. The first exclusion reason wins. Output rules
run only for accepted rows and return exactly their declared new columns.
Callbacks receive pandas frames on both supported model engines, with an ordered
RangeIndex that must be preserved. The rule name, version, function path,
parameters and output dtype are captured with the model. Keep callbacks deterministic
and independent of mutable network, clock or filesystem state; use captured assets
for lookup data. Eligibility runs before temporal preprocessing: excluded rows do
not enter lag/rolling history, and an all-excluded batch retains the previous tail.
Input schema validation still applies to every request.

With the shown source, the expected eligibility is:

| customer_id | Expected status | Reason | Prediction/business output |
| --- | --- | --- | --- |
| 101 | predicted | null | Saved model result; risk_band follows its probability. |
| 102 | excluded | outside_range:age | Null. |
| 103 | excluded | missing:age | Null. |
| 104 | predicted | null | Saved preprocessing fills income and handles NEW; band depends on the fitted model. |

The band callback itself maps probabilities `[0.2, 0.3, 0.7]` to
`[low, medium, high]`; values on a threshold enter the upper band. This is a
fixed callback example, not a measured model prediction. A rule exception fails
the batch; it does not turn an implementation error into an exclusion reason.

## 3. Preview, train and score

After editing the future model's rules, run from the project root:

```powershell
python src/tools/smoke.py
python src/tools/preview.py --action train
python src/tools/preview.py --action score
```

Smoke checks structure; preview resolves the editable declarations and trusted
factories. Neither reads the source table, loads a UC model, runs Spark workers
or validates cloud permissions. Train and inspect the new version using the
[training workflow](databricks_bundle_walkthrough.md), then update model_version
or approve/select the desired champion. Use a fresh prediction-table name when
introducing scoring-status/business columns into a deployment with an older
output schema.

Deploy matching configuration and code, then score:

```powershell
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run score -t test --profile YOUR_PROFILE
```

To select a version only for this run, replace `7` with a real version:

```powershell
databricks bundle run score -t test --profile YOUR_PROFILE --params score_model_version=7
```

The override belongs to the score job. It is not a train-job parameter and does
not mutate inference.yml. A saved-model run uses its captured preprocessing and
scoring rules, even if the editable project now contains different functions.

## 4. Open the correct task and inspect the output

All three layouts use the same default score task keys. The notebook differs:
single_model and model_competition call `src/jobs/score.py`; multi_target calls
`src/jobs/score_models.py`.

| Generated graph | Task keys / where to inspect |
| --- | --- |
| Default recovery flag false | `score`, `monitoring_allowed`, `prepare_monitoring`, `monitoring_ready`, `monitor_model`. Open **score** for prediction results; there is no scoring_report task. |
| Recovery flag true | Adds `recovery_needed`, `recover_predictions`, `scoring_report`. Open **scoring_report** for the ordinary-or-recovery final result. |
| Monitoring tail | monitoring_allowed permits the shared production deployment path; development runs skip that tail. prepare_monitoring validates scoring evidence; monitoring_ready controls the separate monitoring-job handoff. |

The result summary reports the selected concrete model, source end version, input/output counts,
prediction table Delta version and no-op/write status. When scoring policy metadata is present, the
committed manifest also reports predicted_count and excluded_count. Monitoring failure after
publication does not undo a committed prediction write. For a first pinned candidate, enrollment may
require `monitoring_action=prepare_references`; follow
[monitor enrollment](databricks_monitoring_enrollment.md) and retry the original score identity
rather than deleting predictions or inventing a watermark. A no-op may show the previous write's
model provenance alongside this run's selected model; those are not interchangeable.

For the single-model example, query the resolved output name shown in the job.
This SQL assumes no resource suffix; substitute the actual table otherwise:

```sql
SELECT customer_id, prediction, probability_1, risk_band,
       scoring_status, exclusion_reason, model_name, model_version, run_id
FROM workspace.demo.customer_scores
ORDER BY customer_id;
```

Expected: four outcome rows, two predicted and two excluded. Exact estimates and
bands depend on the saved fitted model. Excluded rows have null prediction and
business columns; retaining their keys records that the source observation was
processed. The first successful write's receipt binds the source identity/end
version and prediction output. An all-excluded increment still commits its
outcomes and advances the watermark. A failed rule or write advances neither.

## Inference settings reference

These fields belong in `config/inference.yml`. Defaults below are the values
project generation writes. The `model_set` mapping exists only in multi_target
projects.

| Field | Type / generated default | Allowed values and effect |
| --- | --- | --- |
| `version` | Integer `1` | Companion-file format; distinct from deployed workflow_contract. |
| `batch_compute` | String `single_node` | single_node or spark; selects whole-frame or admitted distributed batch scoring. |
| `score_source_table` | Three-part name; training source by default | Delta input; deployment table placeholders are expanded. |
| `prediction_table` | Three-part name; project predictions name | Single-model/competition output, with the generated resource suffix. |
| `score_model_selection` | String `pinned_version` | pinned_version or champion for single-model/competition scoring. |
| `model_version` | Positive version string; generated `"1"` when pinned | Required concrete pin for pinned single-model selection; replace with an inspected version. |
| `on_model_change` | String `score_new_rows` | score_new_rows or rescore_all; controls behavior when model/set selection changes. |
| `keep_prediction_generations` | Integer `2`, or `null` | With rescore_all, how many newest `_v<n>` generations to keep after a switch; the active one is always kept. `null` keeps all. |
| `rescore_when_history_expires` | Boolean `false` | Opt into the matching deployed recovery graph for recognized expiry only. |
| `follow_champion` | Boolean `false` | true or false; must match deployed graph parameters. |
| `spark_udf_env_manager` | String; generated only for Spark | local on serverless or virtualenv on the generated job-cluster route; must match supported compute/dependencies. |
| `spark_udf_prediction_batch_rows` | Integer `10000` for Spark | 1 to 100000 rows per model prediction call, not a whole-source limit. |
| `model_set` | Mapping; multi_target only | Set identity, destination, rules, publication and promotion policy. See the [model set fields](databricks_model_sets.md#set-the-model_set-block). |

## How new rows are discovered

The first successful run scores a pinned snapshot of the source. Later runs use
the committed source Delta watermark and read CDF from the next version through
a pinned upper version. They do not use `MAX(customer_id)` or an application
clock to guess which records are new. Keys identify outcomes; Delta versions
identify the source change window.

For example, after the first four rows have been committed:

```sql
INSERT INTO workspace.demo.customer_score VALUES
  (105, 45.0, 3500.0, 4.0, 'B');
```

Run score again. Expected: one new outcome for 105. With no further source
changes, another run is a no-op and does not append duplicates. Keep the source
and output under their declared ownership: replacing the source table changes
its identity, and manually rewriting the prediction target can invalidate its
receipt. The workflow rejects those changes rather than guessing a watermark.

## Source changes are separate from model changes

| Event | Default behavior | Explicit alternative |
| --- | --- | --- |
| New source INSERT | Score and append that observation. | No special setting required. |
| Selected model version changes | `score_new_rows` leaves old predictions intact and uses the selected version for new inserts. | `on_model_change: rescore_all` starts a versioned generation, preserving the initial scoring boundary. |
| Readable source UPDATE/DELETE | Ordinary single-model/competition insert scorer rejects it; model sets also default to reject. | Model sets alone support `model_set.on_source_change: rescore_all`. |
| Required CDF history expired | Fail with history-loss evidence. | Deploy the optional recognized-expiry recovery graph described below. |

With score_new_rows, one target can contain predictions from several model
versions. Use stored provenance when comparing rows. Merely choosing a different
version with no new source rows does not rescore the old population.

With rescore_all, the logical prediction name is a managed view over a physical
`<prediction_table>_v<version>` generation. A new version scores the eligible
snapshot and switches the view after successful publication. Then older
generations are dropped automatically, keeping the newest
`keep_prediction_generations` (default `2`: the active one and the one before it,
for rollback) and always the active one. Only `<prediction_table>_v<n>` tables
tagged as this model's full-rebuild output are dropped; set
`keep_prediction_generations: null` in `config/inference.yml` to keep them all.
Repeated runs for that generation can continue incrementally.
With `full_snapshot`, every current source row is eligible. With `from_now`,
the rebuild uses keys already in the previous prediction generation plus new
CDF inserts after its source watermark; pre-start history remains excluded.
A version-specific physical table records the model name, version and artifact
digest. Scoring rejects a table at that name when its saved model provenance
differs. A different registered model sharing the same numeric version needs a
different logical prediction name or a new model version.
Do not switch an existing physical append table to a full-rebuild view under
the same name; choose a fresh logical destination and plan the consumer change.
Full rebuild is a model-selection policy, not a blanket permission to accept
updates in the current generation.

For a multi_target project, merge this **fragment** into its existing model_set
mapping, keeping its registered set and destination settings:

```yaml
model_set:
  on_source_change: rescore_all
```

When CDF contains UPDATE/DELETE, that policy rebuilds the eligible current source
with the selected set, preserving any `from_now` boundary. Deleted observations disappear and corrected values,
business outputs and temporal history are recomputed. Even if append mode had
mixed earlier versions, the source-triggered rebuild uses the selected set for
all eligible rows. It does not retrain. The selected local snapshot must fit row/byte
limits. Do not add on_source_change at the top level or to a single-model
project expecting the same behavior.

## Model-set outputs and business views

A multi_target score loads one saved set with its component versions. Component rules run first;
combined rules then use named results such as `revenue__prediction`. `model_set.publication.mode`
chooses `all`, `combined_only` or `separate_views`. The
[model set guide](databricks_model_sets.md#choose-the-output-mode) owns the modes, view names and
combined rules.

## Recover recognized CDF expiry

`rescore_when_history_expires: true` is opt-in and changes the generated task graph.
Generate/deploy a matching project with recovery enabled; changing inference.yml
alone is rejected because deployed_auto_rebuild_on_cdf_expiry in the job must
agree in both job definitions. Leave recovery disabled until those tasks are
deployed. Keep `workflow_contract: "3"` fixed; it is not the YAML version.

The score task recognizes a supported missing-history condition and emits a
recovery request. recovery_needed selects recover_predictions, which scores the
pinned eligible source snapshot with the same concrete model/set selection. For
`from_now`, recovery excludes keys in the original starting snapshot. That
snapshot must still be readable; if its history or files have expired, recovery
stops rather than scoring pre-start history. Normal model-change rebuilds use
the saved prediction keys and recent CDF instead. A
successful replacement verifies source/output identities and the expected output
version, then atomically retains the output table identity, grants and
named views, then ordinary incremental scoring resumes at the new watermark. The
previous model-version generation remains available (see
`keep_prediction_generations`); rollback inside the rebuilt generation depends on
Delta history retention.
An exact committed recovery retry is a no-op. Open scoring_report for the
completed result; the initial score output saying recovery was requested is not
proof that replacement finished.

This mechanism handles history expiry, not permissions, connectivity, disabled
CDF, source-table replacement, schema mismatch or readable UPDATE/DELETE events.
Those retain their existing failure behavior. Recovery can read the full source,
so keep capacity limits appropriate. It does not retrain or change aliases.
If scoring fails after a successful promotion, repair/retry scoring separately;
do not rerun a lifecycle action whose alias effects may already have committed.

## Schedules, handoff and execution engines

`follow_champion: false` is the default. `true` starts the score
job after successful champion initialization/promotion/rollback. Pending review,
failed quality gates and rejection do not trigger it. It is a deploy-time setting,
not a Run now parameter. To change it, edit inference.yml and the matching
`deployed_score_handoff` value in every `resources/*.job.yml` that carries it
(train, score and monitoring), then redeploy. Changing inference.yml alone is
rejected as a mismatch.
The setup's scoring_mode chooses manual or scheduled **job execution**; it is
separate from `mode` in config/scoring.yml, which chooses eligibility.

Inspect the handoff child score run before launching another score; a separate
invocation is only needed for new rows or recovery.

Schedule, timezone, pause status, score retries, task/job time limits and
notifications live under deployment configuration. Generated defaults do not
add recipients or retries. Keep one owning writer for each prediction target;
job serialization is not a lock against unrelated external writers.

## Distributed inference settings

<span id="worker-configuration-and-admission"></span>
<span id="what-spark-scoring-supports"></span>
<span id="capacity-and-worker-environment"></span>
<span id="certified-estimators-and-model-set-composition"></span>
<span id="results-and-model-set-composition"></span>

Without `batch_compute`, scoring is whole-frame (`single_node`). With
`batch_compute: spark`, the Bundle uses `mlflow.pyfunc.spark_udf` to apply the
saved preprocessing and model to pandas batches on Spark workers; it does not
turn fitting or tuning into distributed training. Admission checks the exact
fitted artifact, and a package logged without a certificate cannot use Spark.
[Score on Spark workers](databricks_spark_scoring.md) covers worker
configuration, the support table, capacity and results.

## Training without publishing a prediction table

The generated `train` and `score` jobs are separate. Set `follow_champion: false`
in `config/inference.yml` to prevent automatic scoring after an alias change.
Choose `scoring_mode: manual` during initialization to omit a score schedule.
For an existing project, also update matching `deployed_score_handoff` parameters
in `resources/*.job.yml`, remove any unwanted score schedule, and redeploy.
The runtime rejects a YAML/job handoff mismatch. Merely configuring
`prediction_table` does not create or populate it; training can still write its
own dataset snapshots and lifecycle evidence. [Endpoint callers](databricks_serving.md)
can consume predictions without publishing a Delta prediction table.

## Common errors and fixes

| Symptom | Action |
| --- | --- |
| No champion / unknown version | Inspect the concrete registered model or set and its approval evidence; use the intended selector/override. |
| Preview rejects a target-reading filter | Decide pre_split/combined skip behavior before training, or explicitly choose custom eligibility. |
| Changing scoring.yml or scoring.py has no effect | Train/package a new version; the selected model retains captured rules. |
| No scoring_status columns | Empty pre_split with no opted-in policy keeps ordinary outputs; select and package an actual policy. |
| Expected scoring_report is missing | Default graph has none; open score. scoring_report belongs to the recovery-enabled graph. |
| UPDATE/DELETE causes failure | Review source correction policy; model-set `on_source_change: rescore_all` is not a single-model option. |
| Source is unchanged but a different model was selected | Incremental append does not rewrite old rows; inspect selected model versus previous write provenance. |
| Existing output schema differs | Use a fresh compatible destination when adding policy/business fields or changing publication shape. |
| Row/byte budget exceeded | Size the first/rebuild population, not only daily inserts; adjust validated capacity or use an admitted distributed route. |
| CDF recovery was requested but no new results appear | Inspect recover_predictions and the final scoring_report; do not treat the request as success. |
