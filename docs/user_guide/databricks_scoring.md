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
| `src/features/scoring.py` | Rules captured with the next model version: reuse pre-split, custom eligibility, business outputs and their callback functions. |
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
inference_mode: local
score_source_table: workspace.demo.customer_score
prediction_table: "{catalog}.{output_schema}.customer_scores{resource_suffix}"
score_model_selection: pinned_version
model_version: "1"
model_change_mode: incremental_append
auto_rebuild_on_cdf_expiry: false
score_handoff: disabled
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
data have different lifecycles. With the same table, the first score includes all
existing rows, including historical labeled observations; confirm that population
before running.

## 2. Decide eligibility before training the version

Scoring rules are saved with the model. Configure them **before training and
packaging the version you will score**. Editing today's scoring.py does not
rewrite an already registered version.

| `SCORING_MODE` | What runs |
| --- | --- |
| `pre_split` (default) | Saved pre-split eligibility; empty recipe leaves the ordinary prediction schema. |
| `custom` | Custom eligibility, model prediction, then custom output rules. |
| `combined` | Saved pre-split eligibility, custom eligibility on its survivors, model prediction, custom outputs. |

With pre_split/combined, a filter reading the target needs an explicit decision:
`SKIP_TARGET_PRE_SPLIT_STEPS = True` skips those filters because scoring has no
actual label. The default false rejects that recipe, including during training
preview. It does not invent labels. A mixed filter reading both target and
features is skipped as a whole, so split the rules when the feature check should
remain. Custom mode ignores this switch.

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

The shipped `scoring.py` eligibility/output lists are empty and its sample rules
are commented out. Activating a mode alone does not enable them. The BEFORE
examples require `feature_value` and a finite range from 0 to 120; the AFTER
example bands predictions below 10, from 10 to below 50, and 50 or above as
low/medium/high. Return `[]` to omit that section. Adapt names and limits, or
use the complete churn example below. Function paths such as
`scoring.require_value_range` name callbacks defined at the end of `scoring.py`;
`params` supplies their inputs without a manual call. A callback receives only the
model's saved input columns; in a model set the same rules run for every component,
so read only columns that every component receives.

### A complete custom-rule example

This example deliberately selects `custom`, so the documented reasons come
from the two custom rules rather than an earlier pre-split exclusion. In
`src/features/scoring.py`, change SCORING_MODE and replace these two existing
functions. Keep the generated build_model_rules/build_scoring wrappers and other
functions. The callback implementations already ship at the end of
`scoring.py`.

```python
SCORING_MODE = "custom"
SKIP_TARGET_PRE_SPLIT_STEPS = True  # Ignored in custom mode.


def build_eligibility_rules():
    """Require an observed age within the declared business range."""
    return [
        {
            "name": "observed_age",
            "version": "1",
            "function": "scoring.require_observed_values",
            "params": {"columns": ["age"]},
        },
        {
            "name": "adult_age",
            "version": "1",
            "function": "scoring.require_value_range",
            "params": {"column": "age", "minimum": 18.0, "maximum": 120.0},
        },
    ]


def build_output_rules():
    """Publish a business band using the saved churn probability column."""
    return [
        {
            "name": "churn_risk_band",
            "version": "1",
            "function": "scoring.prediction_band",
            "params": {
                "column": "probability_1",
                "thresholds": [0.3, 0.7],
                "labels": ["low", "medium", "high"],
                "output": "risk_band",
            },
            "columns": [{"name": "risk_band", "dtype": "string"}],
        },
    ]
```

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
| `inference_mode` | String `local` | local or spark; selects whole-frame or admitted distributed batch inference. |
| `score_source_table` | Three-part name; training source by default | Delta input; deployment table placeholders are expanded. |
| `prediction_table` | Three-part name; project predictions name | Single-model/competition output, with the generated resource suffix. |
| `score_model_selection` | String `pinned_version` | pinned_version or champion for single-model/competition scoring. |
| `model_version` | Positive version string; generated `"1"` when pinned | Required concrete pin for pinned single-model selection; replace with an inspected version. |
| `model_change_mode` | String `incremental_append` | incremental_append or full_rebuild; controls behavior when model/set selection changes. |
| `auto_rebuild_on_cdf_expiry` | Boolean `false` | Opt into the matching deployed recovery graph for recognized expiry only. |
| `score_handoff` | String `disabled` | disabled or after_alias_change; must match deployed graph parameters. |
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
| Selected model version changes | `incremental_append` leaves old predictions intact and uses the selected version for new inserts. | `model_change_mode: full_rebuild` starts a versioned generation from the full source. |
| Readable source UPDATE/DELETE | Ordinary single-model/competition insert scorer rejects it; model sets also default to reject. | Model sets alone support `model_set.source_change_policy: rebuild_on_change`. |
| Required CDF history expired | Fail with history-loss evidence. | Deploy the optional recognized-expiry recovery graph described below. |

In incremental_append, one target can contain predictions from several model
versions. Use stored provenance when comparing rows. Merely choosing a different
version with no new source rows does not rescore the old population.

In full_rebuild, the logical prediction name is a managed view over a physical
`<prediction_table>_v<version>` generation. A new version scores the complete
snapshot and switches the view after successful publication; older generation
tables remain. Repeated runs for that generation can continue incrementally.
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
  source_change_policy: rebuild_on_change
```

When CDF contains UPDATE/DELETE, that policy rebuilds the complete current source
with the selected set. Deleted observations disappear and corrected values,
business outputs and temporal history are recomputed. Even if append mode had
mixed earlier versions, the source-triggered rebuild uses the selected set for
all rows. It does not retrain. The complete local snapshot must fit row/byte
limits. Do not add source_change_policy at the top level or to a single-model
project expecting the same behavior.

## Model-set outputs and business views

A multi_target score loads one saved set with its component versions. Component rules run first;
combined rules then use named results such as `revenue__prediction`. `model_set.publication.mode`
chooses `all`, `combined_only` or `separate_views`. The
[model set guide](databricks_model_sets.md#choose-the-output-mode) owns the modes, view names and
combined rules.

## Recover recognized CDF expiry

`auto_rebuild_on_cdf_expiry: true` is opt-in and changes the generated task graph.
Generate/deploy a matching project with recovery enabled; changing inference.yml
alone is rejected because deployed_auto_rebuild_on_cdf_expiry in the job must
agree in both job definitions. Leave recovery disabled until those tasks are
deployed. Keep `workflow_contract: "3"` fixed; it is not the YAML version.

The score task recognizes a supported missing-history condition and emits a
recovery request. recovery_needed selects recover_predictions, which scores the
pinned full source snapshot with the same concrete model/set selection. A
successful replacement verifies source/output identities and the expected output
version, then atomically retains the output table identity, grants and
named views, then ordinary incremental scoring resumes at the new watermark. Older model-version
generations remain available; rollback inside the rebuilt generation depends on
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

`score_handoff: disabled` is the default. after_alias_change starts the score
job after successful champion initialization/promotion/rollback. Pending review,
failed quality gates and rejection do not trigger it. It is a deploy-time setting,
not a Run now parameter. To change it, edit inference.yml and the matching
`deployed_score_handoff` value in every `resources/*.job.yml` that carries it
(train, score and monitoring), then redeploy. Changing inference.yml alone is
rejected as a mismatch.
The setup's scoring_mode chooses manual or scheduled **job execution**; it is
separate from the Python SCORING_MODE that chooses eligibility.

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

Without `inference_mode`, scoring is whole-frame (`local`). With
`inference_mode: spark`, the Bundle uses `mlflow.pyfunc.spark_udf` to apply the
saved preprocessing and model to pandas batches on Spark workers; it does not
turn fitting or tuning into distributed training. Admission checks the exact
fitted artifact, and a package logged without a certificate cannot use Spark.
[Score on Spark workers](databricks_spark_scoring.md) covers worker
configuration, the support table, capacity and results.

## Training without publishing a prediction table

The generated `train` and `score` jobs are separate. Set `score_handoff: disabled`
in `config/inference.yml` to prevent automatic scoring after an alias change.
Choose `scoring_mode: manual` during initialization to omit a score schedule.
For an existing project, also update matching `deployed_score_handoff` parameters
in `resources/*.job.yml`, remove any unwanted score schedule, and redeploy.
The runtime rejects a YAML/job handoff mismatch. Merely configuring
`prediction_table` does not create or populate it; training can still write its
own dataset snapshots and lifecycle evidence. [Endpoint callers](databricks_serving.md)
can consume predictions without publishing a Delta prediction table.

## Common errors

| Symptom | Action |
| --- | --- |
| No champion / unknown version | Inspect the concrete registered model or set and its approval evidence; use the intended selector/override. |
| Preview rejects a target-reading filter | Decide pre_split/combined skip behavior before training, or explicitly choose custom eligibility. |
| Changing scoring.py has no effect | Train/package a new version; the selected model retains captured rules. |
| No scoring_status columns | Empty pre_split with no opted-in policy keeps ordinary outputs; select and package an actual policy. |
| Expected scoring_report is missing | Default graph has none; open score. scoring_report belongs to the recovery-enabled graph. |
| UPDATE/DELETE causes failure | Review source correction policy; model-set rebuild_on_change is not a single-model option. |
| Source is unchanged but a different model was selected | Incremental append does not rewrite old rows; inspect selected model versus previous write provenance. |
| Existing output schema differs | Use a fresh compatible destination when adding policy/business fields or changing publication shape. |
| Row/byte budget exceeded | Size the first/rebuild population, not only daily inserts; adjust validated capacity or use an admitted distributed route. |
| CDF recovery was requested but no new results appear | Inspect recover_predictions and the final scoring_report; do not treat the request as success. |
