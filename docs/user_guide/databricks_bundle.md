# Local-engine Databricks Bundle

For job-screen instructions and lifecycle diagrams, use the
[operator walkthrough](databricks_bundle_walkthrough.md).

The custom Skyulf template generates one editable Bundle with `dev`, `test`,
`syst` and `prod` targets. It fits and predicts with pandas or Polars. Spark
reads bounded Unity Catalog Delta rows and publishes predictions; local
feature engineering and model prediction are not distributed Spark work.

Initialize a project from a Skyulf checkout:

```powershell
databricks bundle init skyulf-core/templates/databricks --output-dir ./generated
```

Initialization asks for project name, task, engine, existing source tables,
row keys (including composite keys), feature and label columns, model-change
mode, independent scoring and promotion policies, optional
score handoff, retraining mode and cron, serverless
or policy-backed job compute and the existing `dev`
catalog/schema. Serverless is the default. Reviewable
noninteractive examples are in `skyulf-core/templates/databricks/examples/`. The generated
project has its own `databricks.yml`; Skyulf's root has no Bundle config.

Regression starts with Core `linear_regression` and `heldout_rmse`;
classification starts with `logistic_regression` and `heldout_accuracy`.
The pipeline remains editable: add registered Core preprocessing nodes and
choose a task-compatible model. With a JSON init file, `max_rows` and
`max_bytes` are positive integer **strings**; the generated workflow stores
them as numbers. `record_key_columns_json` accepts one or more source key
columns, for example `["customer_id", "observation_id"]`.

### Record identity and result availability

New workflow JSON uses the following names:

```json
{
  "record_key_columns": ["customer_id", "observation_id"],
  "event_column": "observation_date",
  "target_column": "claim_amount",
  "result_available_at_column": "result_confirmed_at"
}
```

The key columns identify a source record and are copied to prediction output.
They do not create an ID, become features, or implicitly define CV groups.
`event_column` names the source observation timestamp; `start`, `holdout_start`
and `cutoff` are boundaries applied to that column. `result_available_at_column`
names each row's actual result-availability timestamp. Different rows can have
different availability dates; the existing temporal workflow excludes results
that are unknown or become available after its cutoff. Skyulf does not invent
those timestamps or fill them with the job's execution time.

Use `record_key_columns_json` and `result_available_at_column` in initialization
files. The default example column names are `entity_id` and `label_at`;
initialization does not create those columns or generate timestamps.

The same names are used throughout the library and saved training settings.
This is an intentional pre-production breaking rename: regenerate projects and
retrain test models created with the previous column settings. There is no
field-alias adapter or automatic conversion of earlier training evidence.

Temporal fields are still required for training. Date-free training and
configurable split/CV are separate pre-SM-34 tasks; do not supply invented dates.

### Source date formats and timezones

Column names and time boundaries serve different purposes. `event_column`
selects the source column; `event_time_parsing` explains its values. The
`start`, `holdout_start` and `cutoff` boundaries are timezone-aware instants.
They do not need to use the same UTC offset as source values.

Edit these settings in the generated `config/workflow.json`:

```json
{
  "event_column": "observation_date",
  "event_time_parsing": {
    "format": "%d/%m/%Y %H:%M:%S",
    "timezone": "Europe/Copenhagen",
    "date_only": "reject"
  },
  "result_available_at_column": "result_confirmed_at",
  "result_time_parsing": {
    "format": "%Y-%m-%dT%H:%M:%S%z",
    "timezone": null,
    "date_only": "reject"
  },
  "start": "2026-06-01T00:00:00+00:00",
  "holdout_start": "2026-08-01T00:00:00+00:00",
  "cutoff": "2026-09-01T00:00:00+00:00"
}
```

In this example `01/08/2026 02:00:00` in Copenhagen and
`2026-08-01T03:00:00+03:00` represent the same instant: August 1 at 00:00 UTC.
An event at that instant belongs to the holdout. The window includes `start`
and excludes `cutoff`; the holdout includes `holdout_start`.

| Source values | Parsing settings |
| --- | --- |
| Native Spark `TIMESTAMP` instants | Keep defaults: `format: null`, `timezone: null`, `date_only: "reject"`; the stored instant is retained. |
| Native `TIMESTAMP_NTZ` or naive local datetimes | Set the source IANA `timezone`; leave `format: null`. |
| Strings containing UTC offsets | Set an explicit `format` with `%z`; leave `timezone: null`. |
| Strings containing local clock times | Set an explicit `format` and source `timezone`. |
| Native `DATE` or date-only strings | Set `date_only: "midnight"` and source `timezone`; strings also need a date `format`. |

Formats use Python numeric directives: `%Y`, `%m`, `%d`, `%H`, `%M`, `%S`,
`%f` and `%z`, with literal separators. A complete four-digit year, month and
day are required. They are not Spark/Java patterns such as `yyyy-MM-dd`.
The same parser handles formatted source values and local validation, so
pandas and Polars training receive the same instants. Precision is limited to
microseconds; submicrosecond pandas timestamps are rejected. Local month names,
missing years and guessed day/month order are not supported. An explicit
`%d/%m/%Y` resolves day/month order; a string without a format does not.

Date-only values are not automatically interpreted as UTC. For example,
`2026-08-01` with `date_only: "midnight"` and `Europe/Copenhagen` means
July 31 at 22:00 UTC, before the holdout boundary above. Ambiguous or nonexistent
local times at daylight-saving transitions fail; supply an offset-bearing
source timestamp to make the intended instant explicit.

Source dates are normalized in Spark before filtering. Invalid/null event
values fail validation even if a window filter would otherwise hide them.
A null result-availability value remains unavailable, while a malformed value
fails. Normalized instants cross the Spark/Python boundary as integer
microseconds; session and process timezone settings do not reinterpret them.
The local row/byte limits still apply after filtering. Date validation may scan
the pinned source snapshot in Spark; these limits bound driver materialization,
not the amount of distributed source validation.

Keep driver, worker and replay environments' timezone data aligned. Saved rules
record zone identifiers, not a copy of the IANA timezone database.

Parsing rules are saved with the candidate's training specification and dataset
identity. Approval replays those saved rules. The job's cron timezone only
controls when it runs; it does not define the source timestamps' timezone.
For Python APIs, use `TrainingDateSpec` in the
[local training guide](databricks_local_sdk.md#train-a-candidate-when-labels-are-ready).
The supported format vocabulary is a subset of Python's
[strptime directives](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes);
source timezone rules use [IANA zoneinfo](https://docs.python.org/3/library/zoneinfo.html).

Manual `training_version`, `start`, `holdout_start` and `cutoff` default to
`null`. Set an actual Delta snapshot and timezone-aware split before running
`train`. Scoring and saved-evidence actions do not require these dates.
`train_monthly` keeps its existing rolling-window behavior; detailed schedule
and window controls are a separate follow-up.

### Configuration validation and migration

New projects declare `config_version: 1` and `task`. The notebook validates
the resolved configuration before training, registry access or output writes:
resource names, distinct columns, reserved metadata names, budgets, model/task
compatibility, metric/threshold domains and action-specific snapshot inputs.
Actual source schema, keys and artifact compatibility are checked by the
existing data-read and scoring services against the real data.

The same check can run locally, without Spark or registry access:

```python
from skyulf.integrations.databricks.local_workflow import resolve_target_config
from skyulf.integrations.databricks.workflow_config import validate_workflow_config

resolved = resolve_target_config(config, {
    "catalog": "workspace",
    "input_schema": "my_inputs",
    "output_schema": "my_outputs",
    "metadata_schema": "my_models",
    "resource_suffix": "",
})
validate_workflow_config(resolved, action="train")  # Or score, approve, etc.
```

For an older project's loaded JSON, migrate explicitly:

```python
from skyulf.integrations.databricks.workflow_config import migrate_workflow_config

updated = migrate_workflow_config(
    config, task="regression", score_handoff="after_alias_change"
)
```

Review/save `updated`, regenerate both job definitions and entrypoints, then
validate and redeploy together. Migration returns a copy; it does not upload
files, invent training dates, or change aliases. Legacy `auto_champion` maps
to champion/automatic; legacy `pinned_version` maps to pinned/manual approval.
Mixed policies, unknown config versions and contradictory existing task or
handoff settings are rejected. The notebook checks the deployed
`workflow_contract` and `deployed_score_handoff` markers against the JSON.
These detect stale generated definitions; they are not workspace permissions
or a complete audit of manually edited Jobs settings.

`dev` uses the selected CLI profile's workspace host. `test`, `syst` and
`prod` each have a different placeholder host and catalog in the generated
file. Edit each target's host, catalog and input/output/metadata schema
bindings, then validate with its designated profile. No Danske, Danica or
personal-workspace value is built into those three targets. Policy-backed
compute uses the policy name, runtime, node type and cost tag supplied at
initialization. Serverless compute needs none of those fields.

For a company policy requiring `PayingRegNo`, select that value as
`cost_tag_key` and supply the approved registration number as `cost_tag_value`.
The `paying-reg-no-init.example.json` example demonstrates both settings;
the generic default remains `CostCenter`. This tag applies to policy-cluster
compute. Selecting it does not assign a model risk category.

Initialization also asks for an optional `risk_category`, such as `Low`,
`Medium`, or `High`. Leave it blank to omit the tag. The value remains editable
in `config/workflow.json` and is recorded on future training runs and model
versions; it does not change promotion gates or relabel existing versions.

## Where the workflow lives

The generated `src/workflow.py` and `src/score.py` are small entrypoints with
fixed lifecycle and score roles. The installed `job_runtime` adapter reads
widgets/configuration, validates operator inputs and publishes task values.
It delegates target resolution and execution to `local_workflow`; training,
evaluation, registry changes and prediction use existing Core services.
`prediction_output` creates output tables and safely switches full-rebuild
views. Imports do not create a Spark session or cloud resource.

Edit the business pipeline in `config/workflow.json`. Model/preprocessing
choices remain project configuration; common workflow fixes ship in the
Skyulf wheel instead of requiring edits to every generated notebook.

### Independent scoring and promotion policies

Generated Bundles and direct `run_action` callers separate these decisions:

| `score_model_selection` | `promotion_policy` | Behavior |
| --- | --- | --- |
| `pinned_version` | `manual_approval` | Register/evaluate the contender; score keeps the configured version |
| `pinned_version` | `automatic` | Apply promotion gates; score still keeps the configured version |
| `champion` | `manual_approval` | Register/evaluate the contender; score follows the existing champion |
| `champion` | `automatic` | Apply promotion gates; the next score follows the resulting champion |

For example, a library configuration containing `model_version: "2"`,
`score_model_selection: "pinned_version"` and `promotion_policy: "automatic"`
continues scoring with v2 after a successful promotion of v4. Selecting
`champion` instead resolves the controlled alias once per score run. A missing
champion fails before output preparation; there is no fallback to latest.

Both new fields are required together. Remove `model_selection_mode` when
using them; mixed or partial configurations fail before training or scoring.
Legacy `auto_champion` retains champion/automatic behavior, and legacy
`pinned_version` retains pinned/manual behavior, with a deprecation warning.
Automatic promotion still requires `quality_threshold`, including when
scoring is pinned. Scoring never updates the caller's configured version.

New Bundles require both policy fields and `score_handoff`. Migrate an older
project's configuration, both notebook entrypoints and job graph together;
changing only JSON is insufficient. Core direct callers retain the legacy
compatibility path described above. Generated projects use the new policies.

### Operator actions through Run now

The existing `train` job is the serialized lifecycle writer. In **Run now with
different parameters**, choose `lifecycle_action`:

| Action | Required job parameters |
| --- | --- |
| `train` / `train_monthly` | No operator evidence; leave other parameters empty |
| `approve` | `candidate_version`, `expected_champion_version`; comparison proof resolves automatically |
| `reject` | Same as approve, plus `rejection_reason` |
| `rollback` | `promotion_receipt_json`, `expected_champion_version` |

Manual training returns copyable `next_actions.approve` and `next_actions.reject`
fields. Review the comparison and copy its exact values; add a reason for
rejection. `expected_champion_version=none` explicitly selects first-champion
initialization; an empty value fails. Configure the metric and absolute quality
threshold **before training** so saved evidence matches the approval policy.
A completed promotion returns `next_actions.rollback`, including its complete
receipt JSON. Copy that receipt rather than reconstructing it from alias names.
These operator actions reuse the existing candidate; they do not fit a model.

`score_handoff=after_alias_change` requests the existing score job after a
successful initialization, promotion or rollback. `disabled` leaves scoring to
an explicit score run or its schedule. Rejection and training without a champion
transition never trigger score. Handoff preserves the score selector: a pinned
scorer still uses its configured version even after champion changes.

There are still two jobs. The lifecycle job contains three tasks: execute the
action, check `score_requested`, and conditionally call the score job. Both jobs
queue runs with `max_concurrent_runs: 1`. Score's fixed notebook role ignores inherited
lifecycle parameters and cannot dispatch alias actions; this is input isolation, not a replacement for exclusive
registry write permissions. A failed/uncertain alias action never publishes a
successful score request. If the alias change succeeded but score failed, retry
score directly; no retraining is needed.

Local tests and strict generated-project validation cover this wiring. The
personal serverless SM-32 rehearsal verified manual approval, rejection,
rollback/retry, automatic promotion, score handoff and pinned selection.
The score entrypoint filters inherited parent-job evidence before dispatching
score. Company targets and production identities remain separate acceptance work.

### Approve an existing candidate without training

Use an explicit manual policy and the comparison returned by the earlier
training job. Review that saved result before passing its version and digest:

```python
import hashlib
import json
from dataclasses import asdict

from skyulf.integrations.databricks.local_workflow import run_action

# candidate is the result of an earlier training job; no training runs here.
comparison = asdict(candidate.comparison)
comparison_sha256 = hashlib.sha256(
    json.dumps(comparison, sort_keys=True, allow_nan=False).encode()
).hexdigest()
receipt = run_action(
    spark,
    config,  # promotion_policy="manual_approval" and both independent fields
    "approve",
    candidate_version=candidate.model_version,
    comparison_sha256=comparison_sha256,
    expected_champion_version=candidate.comparison.champion_version,
)
```

Approval downloads `candidate_comparison.json` and `candidate_training_spec.json`
from the run associated with that registered version. It verifies the digest,
current metric policy, expected champion and controlled challenger receipt.
It then reads the original Delta version and evaluation window and rechecks
quality using the existing Core comparison/promotion services. The original
engine is preserved; current read budgets may tighten the saved limits.

The action does not train, register or upload a model, rewrite `model_version`,
or publish predictions. A later score run follows the configured selector.
Retrying an unchanged successful approval returns its original committed
receipt; changed champion, pending writes or incompatible evidence fail.
Training runs created before the saved specification was introduced cannot
use this action without explicitly supplying that missing provenance through
a separately reviewed migration; there is no fallback to current training dates.
Use the same externally serialized lifecycle writer as training. The Bundle
operator parameters above provide that path with optional score handoff.

### Reject a candidate or roll back a promotion

To decline the candidate above, use the same version and evidence digest with
an explicit manual policy. This is an alternative to approving that candidate:

```python
decision = run_action(
    spark,
    config,
    "reject",
    candidate_version=candidate.model_version,
    comparison_sha256=comparison_sha256,
    expected_champion_version=candidate.comparison.champion_version,
    rejection_reason="Business review requires another candidate",
)
```

Rejection retains both aliases and the evaluation metrics/status. It records
`approval_status=rejected` and the human-readable `approval_reason` separately.
The reason must be nonempty and at most 256 UTF-8 bytes. A rejected version
cannot be implicitly approved, restaged, renominated or made the first champion.
There is no reopen action in this slice; a later training run creates a new candidate.
Neither rejection nor rollback reads training rows, fits or registers a model.

To undo an earlier completed promotion, pass that promotion's receipt:

```python
reversal = run_action(
    spark,
    config,
    "rollback",
    promotion_receipt=receipt,  # kind="promotion", not first-champion initialization
    expected_champion_version=receipt.new_version,
)
```

Rollback is available under either promotion policy. It restores the previous
champion only when the saved transition still matches controlled registry state.
A separately verified challenger is retained. Repeating an unchanged rejection
or rollback returns the same committed receipt; incompatible evidence,
conflicting alias state and unresolved writes stop the action.

All lifecycle actions must use the same externally serialized writer as training.
No additional control table is created. Direct `run_action` calls do not launch
score or rewrite its version pin; the Bundle adapter can request score afterward. A later `champion` score follows the restored alias;
a `pinned_version` score continues using its configured version. Rollback does
not undo predictions already written; scoring's model-change policy still applies.

### Recent challenger history

The Core lifecycle keeps the last displaced contender as `previous_challenger`:

| Action | Champion | Challenger | Previous challenger |
| --- | --- | --- | --- |
| Starting state | v2 | v3 | unset |
| Nominate v4 | v2 | v4 | v3 |
| Repeat nomination of v4 | v2 | v4 | v3 |
| Promote v4 | v4 | unset | v3 |

Only replacing an existing contender rotates this pointer. Evaluation and
promotion retain it, and promotion still puts the old champion in
`previous_champion`. If the historical version becomes champion through
rollback, or becomes challenger again, its history alias is cleared. The
version and its earlier event records remain available.

This alias is neither a complete history nor a scoring fallback. It creates
no table or job. History changes share the lifecycle's pending-event checks
and writer ownership; manual alias/receipt disagreement stops further actions,
including retries. Partial writes require reconciliation before proceeding.
This history extension passed local and personal-serverless Bundle verification,
including replacement, explicit rejection and rollback while retaining a contender.

## What is created

`bundle deploy` creates only two jobs and uploads their code:

| Job | Purpose | UC objects created when run |
| --- | --- | --- |
| `train` | Train/evaluate a candidate or execute approve/reject/rollback | Model/version only for training |
| `score` | Score initial and later CDF rows | Prediction output and rows |

The default project has no schedule. Choosing `monthly_paused` at initialization
adds a paused schedule to the existing `train` job, without adding a third
job or running it at deployment. The Quartz cron and timezone are selected
at initialization and remain editable Bundle variables. No endpoint or Unity Catalog table is
created by deployment alone.
`promotion_policy=automatic` uses the same train job to compare candidate and
champion on a pinned temporal holdout, independently of the score selector. The
selected `metric`, `min_improvement`, and absolute `quality_threshold` stay
editable in `config/workflow.json`. A first champion requires a numeric
absolute threshold because no prior version exists for comparison. Later
versions must pass that threshold and improve on champion by the chosen
minimum. A passing candidate is staged and promoted through checked registry
receipts; an ineligible candidate leaves champion unchanged. The train job
calls the existing score job only after a champion transition when handoff is
enabled. No third job or control table is added.

In both modes, registration nominates `@challenger` before comparison. A tied
or worse candidate retains that alias with `validation_status=rejected` and
a reason; comparison errors retain it with `validation_status=error`.
New contenders replace the pointer without deleting earlier version evidence.
Manual approval records these results without automatically promoting the candidate.
Training resolves the current champion; a stale explicit `champion_version`
fails before fit. Generic Core training remains alias-free unless a caller
supplies the explicit `on_registered` lifecycle callback.

Training runs and model versions share readable metadata: `train_data_version`,
`test_data_version`, `train_data_destination`, `test_data_destination`,
`model_type`, `candidate_date_tag`, and `engine`. Optional `risk_category` is
editable in workflow.json. Since fit and evaluation split one Delta snapshot,
their table names and versions match; `train_start`, `test_start`, and
`data_end` explain the split. The exact dataset identity is saved in
`training_data.json`. Run tags use `task=training`.

Validation reasons use plain English. Technical event tags retain unique IDs
and evidence hashes for rollback/recovery; their JSON fields are `action`,
`from_version`, `proof`, `parent`, `state`, and `previous`. `from_version`
identifies the version previously held by the affected alias, not necessarily
the previous champion. These are audit records,
not settings to edit. Earlier compact receipts remain supported.

With `score_model_selection=champion`, score resolves the alias once to a
concrete version per run, under either promotion policy. Only the serialized
train job identity may write this
model's aliases; other alias-write grants must be removed before training.
Alias promotion and Delta scoring are separate transactions. If scoring fails
after promotion, the last successful prediction output remains and the score
job must be retried. Unknown alias outcomes need receipt reconciliation.
Skyulf marks an alias transition as pending before writing it; automatic
training and scoring stop until that pending event is reconciled. An existing
champion alias set outside this controlled lifecycle also needs reconciliation
before controlled champion scoring or promotion can use it.

For example, a regression project can select its gate in the generated config:

```json
{
  "engine": "polars",
  "score_model_selection": "champion",
  "promotion_policy": "automatic",
  "score_handoff": "after_alias_change",
  "metric": "heldout_rmse",
  "min_improvement": 0.1,
  "quality_threshold": 5.0,
  "model_change_mode": "incremental_append"
}
```

These example values mean RMSE must be at most `5.0`; a later candidate must
reduce champion's RMSE by at least `0.1` on the same holdout. Improvement is
an absolute metric difference, not a percentage. Skyulf derives the direction
from the metric: RMSE/MAE/log loss are minimized, while R²/accuracy/F1 are
maximized. Use a metric supported by the configured model task. The first
comparison reports `reason=no_champion` and `eligible=false` because no
baseline exists; successful bootstrap is recorded separately as
`result.alias_change.kind=initial` in the Bundle output after the absolute gate passes.

If promotion succeeds but scoring fails, the train job reports a failed
dependent score task. Correct the scoring problem and run `score` again;
retraining is unnecessary. Full-rebuild output continues to expose the last
successfully activated generation during this recovery.

The four relevant Unity Catalog names have different roles:

| Name | Meaning |
| --- | --- |
| `training_table` | Existing labeled input table, pinned to a Delta version for training |
| `score_source_table` | Existing CDF-enabled source of rows to predict |
| `prediction_table` | The output table in append mode, or the stable active view in full-rebuild mode |
| `model_name` | Registered Unity Catalog model, not a table |

The first two references point to **the same existing table by default**.
Neither reference creates a table. Separate them only when labeled training
data and new scoring data have different lifecycles. If they stay together,
the first `score` run predicts all existing rows, including historical labeled
rows; review whether that is intended for your use case.

In `full_rebuild`, a version-specific physical table records the model name,
version and artifact digest. Scoring rejects an existing table at that name
when its model provenance differs. A new model with the same numeric version
needs a different logical prediction name or a new model version.

The initialization question `record_key_columns_json` accepts `["customer_id"]`
or multiple existing key columns. It becomes `record_key_columns` in the generated
configuration, and the same column appears in the prediction table. It must
be non-null, `STRING` or `BIGINT`, and unique across the initial data and all
later inserts. For multiple predictions per customer, edit the generated
configuration to a composite key such as
`"record_key_columns": ["customer_id", "observation_id"]` before deployment. The Bundle
does not create an ID in the source. Keep keys out of `input_columns`. With
`customer_id` in the output, a query
can find a customer's predictions:

```sql
SELECT customer_id, prediction, model_version
FROM catalog.schema.predictions
WHERE customer_id = 'C123';
```

## First run

Build and place the matching Skyulf wheel in the generated project's `dist/`,
then edit `config/workflow.json` for real source columns, preprocessing, model,
temporal split and size limits. The JSON values are an example, not a dataset.
Enable Change Data Feed on the scoring source before later inserts arrive.

```powershell
databricks bundle validate --strict -t dev --profile <profile>
databricks bundle deploy -t dev --profile <profile>
databricks bundle run train -t dev --profile <profile>
```

For pinned scoring, inspect the registered version, put that concrete value
in `model_version`, redeploy the changed JSON, then run `score`. For champion
scoring, approve manually or use automatic promotion gates, then run score or
enable the optional handoff. Inspect both lifecycle and score task results. The first score
rejects a missing source, disabled CDF, unsuitable row keys, a model output
mismatch or an existing target schema mismatch before creating prediction
output. It checks initial row count against `max_rows`; each score
also checks decoded transfer bytes against `max_bytes`. Existing tables are
never overwritten. The first score processes the current source
snapshot; later runs process only new inserts since the committed Delta
receipt. A repeat without new rows is a no-op. No monthly date or source
version is entered for each run.

With manual approval, the first candidate requires explicit approval to become
champion. Pinned scoring can use a registered version without that approval. At initialization, choose
`incremental_append` to keep
v1 predictions and score only later source inserts with pinned v2. Choose
`full_rebuild` to write a new physical `<prediction_table>_v2` generation,
switch the stable `prediction_table` view after a successful complete score,
and append future inserts to v2. The previous generation remains available;
a failed rebuild leaves the active view unchanged. Full mode needs view
creation/ownership privileges and cannot silently turn an existing append-mode
physical table into a view. `max_concurrent_runs: 1` serializes the generated score job, but
does not coordinate other jobs. Grant prediction-table writes only to this
job's identity. For multiple publishers, use Skyulf Core's shared admission
provider. Online endpoints and Spark-native FE/model execution remain
separate work.

The optional monthly `train` schedule defaults to 03:00 UTC on day three and
starts paused. Edit its Bundle cron and timezone variables for the desired
monthly run time. After configuring real labeled data and verifying a manual
run, unpause it deliberately. Each run pins the source's latest Delta version and uses the
first day of the current UTC month as the label cutoff. The preceding month is
holdout; `monthly_lookback_months` (default four) controls the full window.
Only labels available by the cutoff are eligible. `@champion` is resolved to a
concrete version for comparison if present. Manual approval leaves champion
unchanged; automatic promotion applies its metric gates. Score handoff follows
only a successful champion transition when enabled. The source version is
pinned at run start, so `label_at` must faithfully record availability.

The older SM-20a personal serverless rehearsal passed, but its jobs and test
schemas were removed at the user's request. The subsequent clean generic
`dev` rehearsal trained a Polars model from 600 real taxi rows, wrote 600
initial and 50 later predictions to one table, and replayed without another
commit. The later two-job design passed a separate personal serverless
rehearsal: 650 existing source rows, one later insert, and a no-op replay
left one prediction table with 651 rows. The `test`, `syst` and `prod`
placeholders have not been deployed in a company workspace.
